import pickle
from rdkit import Chem
import numpy as np  # Assuming trajectory data is compatible with numpy for ease of handling
import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
import numpy as np
import os
import numpy as np

def load_pkl_file(pkl_file_path):
    """Load molecular data from a pickle file."""
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def convert_to_rdkit_mol(data):
    """Extract 'rdmol' objects from the data."""
    rdkit_mols = [data_element['rdmol'] for data_element in data]
    return rdkit_mols

def save_mols_as_sdf(rdkit_mols, output_file_path):
    """Save RDKit molecule objects as an SDF file."""
    writer = Chem.SDWriter(output_file_path)
    for mol in rdkit_mols:
        writer.write(mol)
    writer.close()

def extract_atom_types(mol):
    """Extract atom types from an RDKit molecule object."""
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atom_types

def save_trajectory_as_xyz(trajectory, output_file_path, atom_types):
    """Save molecular trajectory data as an XYZ file for multiple atoms."""
    num_atoms = len(atom_types)
    num_steps = len(trajectory) // num_atoms  # Assuming trajectory is flattened
    
    with open(output_file_path, 'w') as file:
        for step_idx in range(num_steps):
            file.write(f"{num_atoms}\n")
            file.write(f"Step {step_idx + 1}\n")
            for atom_idx in range(num_atoms):
                position_idx = step_idx * num_atoms + atom_idx
                position = trajectory[position_idx]
                atom_type = atom_types[atom_idx]
                x, y, z = position
                file.write(f"{atom_type} {x:.4f} {y:.4f} {z:.4f}\n")


def save_trajectory_as_dcd(trajectory, output_file_path, atom_types):
    """Save molecular trajectory data as a DCD file for multiple atoms.
    
    Args:
        trajectory (np.ndarray): The trajectory data with shape (num_steps, num_atoms, 3).
        output_file_path (str): The path to save the DCD file.
        atom_types (list): A list of atom types for each atom in the molecule.
    """
    num_atoms = len(atom_types)
    num_steps = len(trajectory)

    print(num_atoms, num_steps)

    # Create a Universe with no topology and positions set to zeros
    u = mda.Universe.empty(n_atoms=num_atoms, trajectory=True)
    u.add_TopologyAttr('type', atom_types)

    with DCDWriter(output_file_path, n_atoms=num_atoms) as W:
        for step_idx in range(num_steps):
            # Select the positions for the first num_atoms atoms from each step
            positions = trajectory[step_idx, :num_atoms, :]
            u.atoms.positions = positions
            W.write(u.atoms)

def save_mol_as_pdb(rdkit_mol, output_file_path):
    """Save an RDKit molecule object as a PDB file."""
    writer = Chem.PDBWriter(output_file_path)
    writer.write(rdkit_mol)
    writer.close()

def generate_pdb_from_rdkit_mol(rdkit_mol, output_file_path):
    """Generate a PDB file from an RDKit molecule object, with 3D coordinates if available."""
    # Ensure the molecule has 3D coordinates; generate them if not
    if not rdkit_mol.GetNumConformers():
        # Use RDKit to generate 3D coordinates; this step is optional and depends on your needs
        AllChem.EmbedMolecule(rdkit_mol)
        AllChem.UFFOptimizeMolecule(rdkit_mol)
    
    with open(output_file_path, 'w') as file:
        file.write(Chem.MolToPDBBlock(rdkit_mol))

def split_trajectory_into_conformers(trajectory, num_conformers, atoms_per_conformer):
    conformers = []
    for i in range(num_conformers):
        start_idx = i * atoms_per_conformer
        end_idx = (i + 1) * atoms_per_conformer
        conformer_trajectory = trajectory[:, start_idx:end_idx, :]
        conformers.append(conformer_trajectory)
    return conformers


def update_mol_conformer_positions(rdkit_mol, conformer_positions):
    """Update an RDKit molecule's conformer positions."""
    if not rdkit_mol.GetNumConformers():
        rdkit_mol.AddConformer(Chem.Conformer(rdkit_mol.GetNumAtoms()), assignId=True)
    conf = rdkit_mol.GetConformer()
    
    expected_atom_count = rdkit_mol.GetNumAtoms()
    #if len(conformer_positions) != expected_atom_count:
    #    print(f"Warning: The number of atoms in the RDKit molecule ({expected_atom_count}) does not match the number of positions provided ({len(conformer_positions)}). Skipping position update.")
    #    return  # Skip updating positions if counts don't match

    for atom_idx in range(expected_atom_count):
        try:
            # Convert x, y, z from numpy.float32 to Python float (double in C++)
            x, y, z = map(float, conformer_positions[atom_idx])
            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
        except IndexError:
            print(conformer_positions)
            print(f"Warning: IndexError encountered for atom index {atom_idx}. This atom's position will not be updated.")

def generate_and_save_pdb_from_conformer(rdkit_mol, conformer_positions, output_file_path):
    """Generate a PDB file for a given conformer."""
    update_mol_conformer_positions(rdkit_mol, conformer_positions)
    pdb_content = Chem.MolToPDBBlock(rdkit_mol)
    with open(output_file_path, 'w') as file:
        file.write(pdb_content)


def process_samples_in_directory(directory_path):
    """Process all 'samples_199.pkl' files in the specified directory, 
    only considering direct child directories starting with 'sample_'."""
    # List all items in the base directory
    for item in os.listdir(directory_path):
        # Construct the full path of the item
        item_path = os.path.join(directory_path, item)
        # Check if this item is a directory and starts with 'sample_'
        if os.path.isdir(item_path) and item.startswith("sample_"):
            # Construct the path to the samples_199.pkl file within this directory
            sample_file_path = os.path.join(item_path, "samples_all.pkl")
            print(f"Looking for file at: {sample_file_path}")  # Debugging statement
            if os.path.exists(sample_file_path):
                # Process each pickle file
                print(f"Found and processing: {sample_file_path}")  # Confirmation of found file
                data = load_pkl_file(sample_file_path)
                process_data(data, item, True)
            else:
                print(f"File not found: {sample_file_path}")  # Warning if file is not found

def sanitize_filename(filename):
    """Sanitize the filename to remove or replace invalid characters."""
    invalid_chars = "<>:\"/\\|?*"
    for char in invalid_chars:
        filename = filename.replace(char, "_")  # Replace invalid chars with underscore
    return filename

def encode_smiles(smiles):
    """Encode a SMILES string into a hexadecimal representation."""
    return ''.join(format(ord(char), 'x') for char in smiles)

def decode_smiles(encoded_smiles):
    """Decode a previously encoded hexadecimal SMILES string."""
    hex_chars = [encoded_smiles[i:i+2] for i in range(0, len(encoded_smiles), 2)]
    return ''.join(chr(int(char, 16)) for char in hex_chars)


def process_data(data, dir_name, no_traj):
    save_final_conformer = True

    for i, data_element in enumerate(data):
        rdkit_mol = data_element['rdmol']
        atom_types = extract_atom_types(rdkit_mol)

        if no_traj:
            pos_ref_shape = np.array(data_element['pos_ref']).shape[0]
            pos_gen_shape = np.array(data_element['pos_gen']).shape[0]
            num_conformers = pos_gen_shape // pos_ref_shape
            atoms_per_conformer = pos_ref_shape // 3
            
            last_positions_segment = np.array(data_element['pos_gen'])[-atoms_per_conformer:]
            
            #sanitized_smiles = sanitize_filename(data_element['smiles'])
            sanitized_smiles = encode_smiles(data_element['smiles'])
            output_pdb_path = f"DRUGS_PLOT/{sanitized_smiles}_{dir_name}_{num_conformers-1}_{i}.pdb"
            generate_and_save_pdb_from_conformer(rdkit_mol, last_positions_segment, output_pdb_path)
    else:
        for i, data_element in enumerate(data):
            print(data_element)
            print(data_element.smiles)
            atom_types = extract_atom_types(data_element['rdmol'])
            trajectory = np.array(data_element['pos_gen'])
            atoms_per_conformer = trajectory.shape[1]
            print(trajectory.shape, len(atom_types))
            input()
            conformer_trajectories = split_trajectory_into_conformers(trajectory, num_conformers, atoms_per_conformer)

            if save_final_conformer:
                for j, conformer_trajectory in enumerate(conformer_trajectories):
                    conformer_positions = conformer_trajectory[-1]
                    print(conformer_trajectory.shape)
                    output_pdb_path = f"DRUGS_PLOT/{data_element.smiles}_{dir_name}_{j}_{i}_.pdb"
                    generate_and_save_pdb_from_conformer(data_element['rdmol'], conformer_positions, output_pdb_path)
            

if __name__ == "__main__":
    base_directory = "logs/drugs_default_2024_03_08__15_12_20/"
    process_samples_in_directory(base_directory)
