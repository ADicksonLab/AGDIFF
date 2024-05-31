import pickle
from rdkit import Chem
import numpy as np  # Assuming trajectory data is compatible with numpy for ease of handling
import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
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

'''def save_trajectory_as_dcd(trajectory, output_file_path, atom_types):
    """Save molecular trajectory data as a DCD file for multiple atoms."""
    num_atoms = len(atom_types)
    num_steps = len(trajectory) // num_atoms

    # Create a Universe with no topology and positions set to zeros
    u = mda.Universe.empty(n_atoms=num_atoms, trajectory=True)
    u.add_TopologyAttr('type', atom_types)

    with DCDWriter(output_file_path, n_atoms=num_atoms) as W:
        for step_idx in range(num_steps):
            # Reshape trajectory to have shape (num_atoms, 3)
            positions = trajectory[step_idx * num_atoms: (step_idx + 1) * num_atoms]
            u.atoms.positions = positions
            W.write(u.atoms)'''


def save_trajectory_as_dcd(trajectory, output_file_path, atom_types):
    """Save molecular trajectory data as a DCD file for multiple atoms.
    
    Args:
        trajectory (np.ndarray): The trajectory data with shape (num_steps, num_atoms, 3).
        output_file_path (str): The path to save the DCD file.
        atom_types (list): A list of atom types for each atom in the molecule.
    """
    num_atoms = len(atom_types)
    num_steps = len(trajectory)

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
        
# Main execution
if __name__ == "__main__":
    pkl_file_path = 'logs/qm9_default_2024_04_24__13_32_23/sample_2024_04_25__12_27_45/samples_all.pkl'
    data = load_pkl_file(pkl_file_path)
    
    for i, data_element in enumerate(data):
        # Generate and save PDB file for topology
        pdb_file_path = f"molecule_{i}.pdb"
        generate_pdb_from_rdkit_mol(data_element['rdmol'], pdb_file_path)
        print(pdb_file_path)

        # Assuming 'pos_gen' is structured as expected for trajectory data
        atom_types = extract_atom_types(data_element['rdmol'])
        trajectory = np.array(data_element['pos_gen'])
        

        conformer1_trajectory = trajectory[:, :1159, :]
        conformer2_trajectory = trajectory[:, 1159:, :]

        dcd_file_path = f"trajectory_1_{i}.dcd"
        dcd_file_path2 = f"trajectory_2_{i}.dcd"

        save_trajectory_as_dcd(conformer1_trajectory, dcd_file_path, atom_types)
        save_trajectory_as_dcd(conformer2_trajectory, dcd_file_path2, atom_types)