import os
import os.path as osp
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
from rdkit.Chem import AllChem
import random

from agdiff.models.epsnet import *
from agdiff.utils.datasets import *
from agdiff.utils.transforms import *
from agdiff.utils.misc import *
from agdiff.models.common import _extend_graph_order
from agdiff.utils.chem import BOND_TYPES


def rdmol_to_data(mol:Mol, smiles=None):
    """
    Converts a Mol object to torch_geometric.data.Data object

    :param mol: Mol object of the molecule.
    :param smiles: smiles strings.
    :return: Data object.
    """
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

    return data


def num_confs(num:str):
    """
    Calculate number of samples

    :param num: number of reference.
    :return: number of samples.
    """
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()


def save_dcd_alanine(pkl_path, pdb_path, dcd_dir):

    """
    Creates DCD trajectories from the sample pkl file.

    :param pkl_path: Path to the saved sample pkl file.
    :param pdb_path: Path to the PDB file.
    :param dcd_dir: Path to directory you want to save DCD files.
    :return: None.
    """

    top = mdj.load(pdb_path).topology
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    pos_gen = data[0].pos_gen / 10  # like [5000, 5500, 3] convert ang --> nm
    num_timesteps, total_atoms, _ = pos_gen.shape
    num_atoms = data[0].pos.shape[0]
    num_trajs = int(total_atoms / num_atoms)
    traj_positions = torch.split(pos_gen, num_atoms, dim=1) 
   
    if not osp.exists(dcd_dir):
        os.makedirs(dcd_dir)
    
    for i, traj_pos in enumerate(traj_positions):
        traj_dcd_path = osp.join(dcd_dir, f'traj_{i}.dcd')
        mdj_traj = mdj.Trajectory(xyz=traj_pos.numpy(), topology=top)
        mdj_traj.save(traj_dcd_path)

        if i ==20: # you may change that to create desired number of DCD files.
            break
        print(f'DCD saved for trajectory {i} at {traj_dcd_path}')
    


def calc_rmsd(traj_id, dcd_dir,pdb_path, output_dir):

    """
    Calculates RMSD of heavy atoms.

    :param traj_id: traj id of te DCD file 
    :param dcd_dir: Path to saved DCDs directory.
    :param pdb_path: Path to PDB file.
    :PARAM txt_dir: Path to the output directory to save calculated RMSDs in txt file.
    :return: None.
    """
    dcd_path = osp.join(dcd_dir, f'traj_{traj_id}.dcd')
    output_path = osp.join(output_dir, f'rmsd_{traj_id}.txt')

    top = mdj.load(pdb_path).topology
    ref_traj = mdj.load(pdb_path)
    traj = mdj.load(dcd_path, top=top)
    print(f"Loaded trajectory with {traj.n_atoms} atoms and {traj.n_frames} frames.")
    
    non_hydrogen_selection = ref_traj.topology.select("not element H")
    ref_traj = ref_traj.atom_slice(non_hydrogen_selection)
    traj = traj.atom_slice(non_hydrogen_selection)
    rmsd = mdj.rmsd(traj, ref_traj)

    with open (output_path , 'w') as f:
        for timestep , rmsd_value in enumerate(rmsd):
            f.write(f"{timestep}\t{rmsd_value:.6f}\n")
    print(f'rmsd values for traj {traj_id} are saved in {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('config' , type = str , help='path for config .yml file')
    parser.add_argument('--save_traj', action='store_true', default=True,
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('5x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5,
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
 
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_random = random.randint(0, 1000000)
    print("Running in SEED: ", seed_random)
    seed_all(seed_random)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # Logging
    # output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    # logger = get_logger('test', output_dir)
    # logger.info(args)

    output_dir = get_new_log_dir(os.path.join(log_dir,"samples"), 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
    ])

    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])

    model.eval()

    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)

    # Load the PDB file
    mol = Chem.MolFromPDBFile("alanine_dipeptide.pdb", removeHs=False) 

    # Ensure the molecule is properly loaded and has a conformer
    if mol is not None and mol.GetNumConformers() > 0:
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

    
    num_refs = 1100 // data_input.num_nodes # you may change num_refs based on molecule
    num_samples = args.num_confs(num_refs)
    print(f'num_refs: {num_refs}')
    print(f'num_samples: {num_samples}')

    data_input['pos_ref'] = None
    batch = repeat_data(data_input, num_samples).to(args.device)

    clip_local = None
   
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

    save_path = os.path.join(output_dir, 'sample.pkl')
    logger.info('Saving samples to: %s' % save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


    # Example of creating DCD and calculating RMSD

    # pkl_path = 'sample_2024_09_04__17_30_25/samples.pkl'
    # pdb_path = 'alanine_dipeptide.pdb'
    # dcd_dir = 'sample_2024_09_04__17_30_25/trajs'
    # output_dir = 'sample_2024_09_04__17_30_25/rmsds'

    # if not osp.exists(txt_dir):
    #     os.makedirs(txt_dir)

    # if not osp.exists(dcd_dir):
    #     os.makedirs(dcd_dir)

    # save_dcd_alanine(pkl_path, pdb_path, dcd_dir)

    # for i in range(20): 
    #     calc_rmsd_alanine(traj_id = i, dcd_dir = dcd_dir, pdb_path = pdb_path, output_dir = output_dir)
   

 