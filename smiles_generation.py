import argparse
import copy
import os
import pickle
import random
from glob import glob

import numpy as np
import torch
import yaml
from easydict import EasyDict
from models.common import _extend_graph_order
from models.epsnet import get_model
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data
from torch_geometric.utils import (dense_to_sparse, remove_self_loops,
                                   to_dense_adj)
from torch_scatter import scatter
from torch_sparse import coalesce
from tqdm.auto import tqdm
from utils.chem import BOND_TYPES
from utils.datasets import PackedConformationDataset
from utils.misc import get_new_log_dir, repeat_data, seed_all
from utils.transforms import AddHigherOrderEdges, Compose, CountNodesPerGraph


def num_confs(num: str):
    print(f"Parsing num_confs argument: {num}")
    if num.endswith("x"):
        multiplier = int(num[:-1])
        print(f"num_confs ends with 'x', multiplier set to: {multiplier}")
        return lambda x: x * multiplier
    elif num.isdigit() and int(num) > 0:
        absolute = int(num)
        print(f"num_confs is a positive integer, absolute count set to: {absolute}")
        return lambda x: absolute
    else:
        raise ValueError(f"Invalid num_confs value: {num}")


def rdmol_to_data(mol: Chem.Mol, smiles=None):
    print("Converting RDKit molecule to PyTorch Geometric Data object...")
    N = mol.GetNumAtoms()
    print(f"Number of atoms in molecule: {N}")

    # Initialize positions to zeros since we are not using RDKit's conformer positions
    pos = torch.zeros((N, 3), dtype=torch.float32)
    print("Initialized atomic positions to zeros.")

    # Extract atomic properties
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
    print("Extracted atomic properties.")

    z = torch.tensor(atomic_number, dtype=torch.long)
    print("Atomic numbers tensor created.")

    # Extract bond information using BOND_TYPES_MAPPING from utils.chem
    row, col, edge_type = [], [], []
    BOND_TYPES_MAPPING = {bond_type: idx for idx, bond_type in enumerate(BOND_TYPES)}
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type = bond.GetBondType()
        bond_type_idx = BOND_TYPES_MAPPING.get(bond_type, 0)
        edge_type += 2 * [bond_type_idx]
    print("Extracted bond information.")

    # Convert bond information to tensors
    try:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        print("Edge index and bond types tensors created.")
    except Exception as e:
        print(f"Error converting bond information to tensors: {e}")
        raise

    # Sort edge indices and bond types
    try:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        print("Sorted edge indices and bond types.")
    except Exception as e:
        print(f"Error sorting edge indices and bond types: {e}")
        raise

    # Coalesce to ensure unique edges
    print("Coalescing edge_index and edge_type to ensure unique edges...")
    try:
        edge_index, edge_type = coalesce(edge_index, edge_type, N, N)
        print(
            f"After coalesce: edge_index shape: {edge_index.shape}, edge_type shape: {edge_type.shape}"
        )
    except Exception as e:
        print(f"Error during coalesce of edge_index and edge_type: {e}")
        raise

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
        print(f"Generated SMILES from molecule: {smiles}")

    # Create PyTorch Geometric Data object without including 'rdmol' to avoid segmentation faults
    try:
        data = Data(
            atom_type=z,
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            smiles=smiles,
        )
        print("Data object created successfully.")
    except Exception as e:
        print(f"Error creating Data object: {e}")
        raise

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate conformers from SMILES and save as SDF."
    )
    parser.add_argument("ckpt", type=str, help="Path for loading the checkpoint")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES string")
    parser.add_argument(
        "--out_sdf", type=str, required=True, help="Output SDF file path"
    )
    parser.add_argument(
        "--save_traj",
        action="store_true",
        default=False,
        help="Whether to store the whole trajectory for sampling",
    )
    parser.add_argument(
        "--num_confs",
        type=num_confs,
        default=num_confs("5x"),
        help='Number of conformers to generate. Use "Nx" to multiply, e.g., "2x"',
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Tag for the output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help='Device to use: "cuda" or "cpu"'
    )
    parser.add_argument(
        "--clip", type=float, default=1000.0, help="Clipping value for gradients"
    )
    parser.add_argument(
        "--n_steps", type=int, default=5000, help="Number of sampling steps"
    )
    parser.add_argument(
        "--global_start_sigma",
        type=float,
        default=0.5,
        help="Enable global gradients only when noise is low",
    )
    parser.add_argument(
        "--w_global", type=float, default=1.0, help="Weight for global gradients"
    )
    parser.add_argument(
        "--sampling_type",
        type=str,
        default="ld",
        help="Sampling method: generalized, ddpm_noisy, ld",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Weight for DDIM and DDPM: 0->DDIM, 1->DDPM",
    )
    args = parser.parse_args()

    print("Starting generate_conformer.py script...")
    print(f"Parsed arguments: {args}")

    # Load checkpoint
    print("Loading checkpoint...")
    try:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        print(f"Checkpoint loaded successfully from {args.ckpt}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    # Load configuration
    config_path = glob(
        os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), "*.yml")
    )
    if len(config_path) == 0:
        print("Configuration YAML file not found.")
        exit(1)
    config_path = config_path[0]
    print(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, "r") as f:
            config = EasyDict(yaml.safe_load(f))
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    # Set random seed for reproducibility
    print("Setting random seed...")
    try:
        seed_random = random.randint(0, 1000000)
        print("Running with SEED: ", seed_random)
        seed_all(seed_random)
    except Exception as e:
        print(f"Error setting random seed: {e}")
        exit(1)

    # Determine log/output directory
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))
    output_dir = get_new_log_dir(log_dir, "sample", tag=args.tag)
    print(f"Output directory is set to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created or already exists: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        exit(1)

    # Model loading
    print("Loading model...")
    try:
        model = get_model(ckpt["config"].model).to(args.device)
        print("Model instantiated successfully.")
        model.load_state_dict(ckpt["model"])
        print("Model state loaded successfully.")
        model.eval()
        print("Model set to evaluation mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Process the SMILES string
    smiles = args.smiles
    print(f"Processing SMILES string: {smiles}")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        print("SMILES string parsed into RDKit molecule successfully.")
    except Exception as e:
        print(f"Failed to parse SMILES string '{smiles}': {e}")
        exit(1)

    # Add hydrogens
    print("Adding hydrogens to the molecule...")
    try:
        mol = Chem.AddHs(mol)
        print("Hydrogens added to the molecule successfully.")
    except Exception as e:
        print(f"Error adding hydrogens: {e}")
        exit(1)

    # Generate an initial 3D conformer
    print("Generating initial 3D conformer...")
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed_random  # Use the same random seed for reproducibility
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            raise ValueError("Conformer generation failed.")
        print("Initial 3D conformer generated successfully.")
    except Exception as e:
        print(f"Failed to generate initial conformer: {e}")
        exit(1)

    if mol.GetNumConformers() == 0:
        print("No conformers found in the molecule after embedding.")
        exit(1)
    else:
        print(f"Number of conformers in the molecule: {mol.GetNumConformers()}")

    # Convert RDKit molecule to data object
    print("Converting molecule to data object...")
    try:
        data_input = rdmol_to_data(mol)
        print("Molecule converted to data object successfully.")
    except Exception as e:
        print(f"Error converting molecule to data object: {e}")
        exit(1)

    # Apply transforms
    print("Applying transforms to data object...")
    transforms = Compose(
        [
            CountNodesPerGraph(),
            AddHigherOrderEdges(
                order=config.model.edge_order
            ),  # Offline edge augmentation
        ]
    )

    try:
        data_input = transforms(data_input)
        print("Transforms applied successfully.")
    except Exception as e:
        print(f"Error applying transforms: {e}")
        exit(1)

    num_nodes = data_input.num_nodes
    print(f"Number of atoms in the molecule: {num_nodes}")

    # Determine number of conformers to generate
    print("Determining number of conformers to generate...")
    try:
        num_refs = 1100 // num_nodes
        num_samples = args.num_confs(num_refs)
        print(f"Number of references (num_refs): {num_refs}")
        print(f"Number of conformers to generate (num_samples): {num_samples}")
    except Exception as e:
        print(f"Error determining number of conformers: {e}")
        exit(1)

    # Prepare batch
    print("Preparing batch data...")
    data_input.pos_ref = None
    try:
        batch = repeat_data(data_input, num_samples).to(args.device)
        print("Data repeated and moved to the specified device successfully.")
    except Exception as e:
        print(f"Error preparing batch data: {e}")
        exit(1)

    # Sampling
    print("Starting conformer sampling...")
    clip_local = None
    success = False
    results = []
    done_smiles = set()
    for attempt in range(2):  # Maximum number of retries
        print(f"Sampling conformations (Attempt {attempt + 1})...")
        try:
            pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
            print("Initialized random positions for sampling.")
            with torch.no_grad():
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False,  # Done in transforms
                    n_steps=args.n_steps,
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                )
            pos_gen = pos_gen.cpu()
            print("Conformations sampled successfully.")

            if args.save_traj:
                data_input.pos_gen = torch.stack(pos_gen_traj)
            else:
                data_input.pos_gen = pos_gen
            results.append(data_input)
            done_smiles.add(data_input.smiles)

            save_path = os.path.join(output_dir, "samples_0.pkl")
            print(f"Saving samples to: {save_path}")
            with open(save_path, "wb") as f:
                pickle.dump(results, f)

            success = True
            break  # Break the retry loop if successful
        except FloatingPointError:
            clip_local = 20
            print("FloatingPointError encountered. Retrying with local clipping.")
        except Exception as e:
            print(f"Error during sampling: {e}")
            exit(1)

    if not success:
        print("Sampling failed after retries.")
        exit(1)

    # Save all samples
    save_path = os.path.join(output_dir, "samples_all.pkl")
    print(f"Saving all samples to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    # Reshape generated positions
    print("Reshaping generated positions...")
    try:
        pos_gen = pos_gen.view(num_samples, num_nodes, 3)
        print(f"Generated positions reshaped to ({num_samples}, {num_nodes}, 3).")
    except Exception as e:
        print(f"Error reshaping generated positions: {e}")
        exit(1)

    # Update RDKit molecule with generated conformers
    print("Updating molecule with generated conformers...")
    try:
        mol.RemoveAllConformers()
        print("Removed all existing conformers from the molecule.")
        for i in range(num_samples):
            conf = Chem.Conformer(num_nodes)
            positions = pos_gen[i]  # Shape: (num_nodes, 3)
            for atom_idx in range(num_nodes):
                x, y, z = positions[atom_idx].tolist()
                conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
            conf.SetId(i)
            mol.AddConformer(conf, assignId=True)
            print(f"Added conformer {i} to the molecule.")
        print(f"Total of {num_samples} conformers added to the molecule.")
    except Exception as e:
        print(f"Error updating molecule with conformers: {e}")
        exit(1)

    # Write the molecule with conformers to an SDF file
    print("Writing conformers to SDF file...")
    try:
        writer = Chem.SDWriter(args.out_sdf)
        for conf in mol.GetConformers():
            writer.write(mol, confId=conf.GetId())
            print(f"Conformer {conf.GetId()} written to SDF.")
        writer.close()
        print(f"All generated conformers saved to {args.out_sdf} successfully.")
    except Exception as e:
        print(f"Error writing to SDF file: {e}")
        exit(1)

    # Optionally save the trajectory
    if args.save_traj:
        traj_save_path = os.path.join(output_dir, "trajectory.pkl")
        print(f"Saving sampling trajectory to {traj_save_path}...")
        try:
            with open(traj_save_path, "wb") as f:
                pickle.dump(pos_gen_traj, f)
            print(f"Sampling trajectory saved to {traj_save_path} successfully.")
        except Exception as e:
            print(f"Error saving sampling trajectory: {e}")
            exit(1)

    print("Conformer generation process completed successfully.")
