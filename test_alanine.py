import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from models.common import _extend_graph_order
from rdkit.Chem import AllChem
import random

from utils.chem import BOND_TYPES

def rdmol_to_data(mol:Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    #data.nx = to_networkx(data, to_undirected=True)

    return data


def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true', default=False,
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('5x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    #seed_all(config.train.seed)
    seed_random = random.randint(0, 1000000)
    print("Running in SEED: ", seed_random)
    seed_all(seed_random)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
    ])
    if args.test_set is None:
        test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
    else:
        test_set = PackedConformationDataset(args.test_set, transform=transforms)

    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])

    model.eval()

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)

    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)
    
    #if data.smiles in done_smiles:
    #    logger.info('Molecule#%d is already done.' % i)
    #    continue

    #data_input = data.clone()

    #batch = repeat_data(data_input, num_samples).to(args.device)

    # Load the PDB file
    mol = Chem.MolFromPDBFile("alanine_dipeptide.pdb", removeHs=False)

    # Ensure the molecule is properly loaded and has a conformer
    if mol is not None and mol.GetNumConformers() > 0:
        # Preprocess the molecule if necessary (e.g., adding hydrogens, generating conformers)
        AllChem.Compute2DCoords(mol)
        
        # Use the rdmol_to_data function
        data_input = rdmol_to_data(mol)

        # Example usage
        num_nodes = data_input.num_nodes
        edge_index = data_input.edge_index
        edge_type = data_input.edge_type

        new_edge_index, new_edge_type = _extend_graph_order(num_nodes, edge_index, edge_type, order=3)

        # Update the data object with extended edges
        data_input.edge_index = new_edge_index
        data_input.edge_type = new_edge_type
    else:
        print("Failed to load the molecule or no conformers found.")


    num_refs = 1100 // data_input.num_nodes
    num_samples = args.num_confs(num_refs)

    data_input['pos_ref'] = None
    batch = repeat_data(data_input, num_samples).to(args.device)

    clip_local = None
    for _ in range(2):  # Maximum number of retry
        try:
            pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
            pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                atom_type=batch.atom_type,
                pos_init=pos_init,
                bond_index=batch.edge_index,
                bond_type=batch.edge_type,
                batch=batch.batch,
                num_graphs=batch.num_graphs,
                extend_order=False, # Done in transforms.
                n_steps=args.n_steps,
                step_lr=1e-6,
                w_global=args.w_global,
                global_start_sigma=args.global_start_sigma,
                clip=args.clip,
                clip_local=clip_local,
                sampling_type=args.sampling_type,
                eta=args.eta
            )
            pos_gen = pos_gen.cpu()
            if args.save_traj:
                data_input.pos_gen = torch.stack(pos_gen_traj)
            else:
                data_input.pos_gen = pos_gen
            results.append(data_input)
            done_smiles.add(data_input.smiles)

            save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
            logger.info('Saving samples to: %s' % save_path)
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

            break   # No errors occured, break the retry loop
        except FloatingPointError:
            clip_local = 20
            logger.warning('Retrying with local clipping.')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
        
    