# test.py
import argparse
# agdif.utils.transform
import copy
import os
import os.path as osp
import pickle as pkl
import sys

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
# agdiff.utils.chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
# agdiff.utils.misc
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import coalesce

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}


###########################################################################
# agdiff.utils.transform
class AddHigherOrderEdges(object):
    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            self.binarize(
                adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)
            ),
        ]

        for i in range(2, order + 1):
            adj_mats.append(self.binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    def __call__(self, data: Data):
        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(
            0
        )  # (N, N)
        type_highorder = torch.where(
            adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order)
        )
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(
            new_edge_index, new_edge_type.long(), N, N
        )  # modify data
        edge_index_1, data.edge_order = coalesce(
            new_edge_index, edge_order.long(), N, N
        )  # modify data
        data.is_bond = data.edge_type < self.num_types
        assert (data.edge_index == edge_index_1).all()

        return data


class CountNodesPerGraph(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


# end agdiff.utils.transform
###########################################################################


###########################################################################
# agdiff.models.common


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices.
        new_edge_type:  Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)  # (N, N)
    type_highorder = torch.where(
        adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order)
    )
    torch._assert(
        (type_mat * type_highorder == 0).all(), "type_mat and type_highorder overlap"
    )

    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)

    new_edge_index, new_edge_type = coalesce(
        new_edge_index, new_edge_type.long(), N, N
    )  # modify data

    return new_edge_index, new_edge_type


# end agdiff.models.common
###########################################################################


def rdmol_to_data(mol: Mol, smiles=None):
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

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = Data(
        atom_type=z,
        pos=pos,
        edge_index=edge_index,
        edge_type=edge_type,
        rdmol=copy.deepcopy(mol),
        smiles=smiles,
    )

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generating Sampling Inputs",
        usage="%(prog)s <smile> <ligand_id> [--out_dir]",
    )

    parser.add_argument("smiles", type=str, help="molecule smile")
    parser.add_argument("ligand_id", type=str, help="molecule ID")
    parser.add_argument("--out_dir", type=str, default="./sampling_inputs")
    parser.add_argument("--ligand_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_edge_order", type=int, default=3)

    args = parser.parse_args()

    transforms = Compose(
        [
            CountNodesPerGraph(),
            AddHigherOrderEdges(order=args.num_edge_order),  # Offline edge augmentation
        ]
    )

    smiles = args.smiles

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    print("after EmbedMolecule")
    for atom in mol.GetAtoms():
        print(f"Atom index: {atom.GetIdx()}, Symbol: {atom.GetSymbol()}")

    if mol is None:
        raise Exception("SMILES was not found in rdkit!")

    # Ensure the molecule is properly loaded and has a conformer
    if mol is not None and mol.GetNumConformers() > 0:
        AllChem.Compute2DCoords(mol)

        # Use the rdmol_to_data function
        data_input = rdmol_to_data(mol)

        num_nodes = data_input.num_nodes
        edge_index = data_input.edge_index
        edge_type = data_input.edge_type

        new_edge_index, new_edge_type = _extend_graph_order(
            num_nodes, edge_index, edge_type, order=args.num_edge_order
        )

        # Update the data object with extended edges
        data_input.edge_index = new_edge_index
        data_input.edge_type = new_edge_type
    else:
        print("Failed to load the molecule or no conformers found.")

    batch = Batch.from_data_list([data_input]).to(args.device)

    atom_types = batch.atom_type.tolist()
    edge_idxs = batch.edge_index.tolist()
    edge_types = batch.edge_type.tolist()

    # Print in AGDIFF_OpenMM input format
    print(f"\nInputs for AGDIFF_OpenMM Sampling:\n")
    print(f"atom_types = {atom_types}\n")
    print(f"edge_idxs = {edge_idxs}\n")
    print(f"edge_types = {edge_types}\n")


out_dir = osp.join(args.out_dir, f"{args.ligand_id}")

if args.ligand_id is not None:
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    with open(osp.join(out_dir, f"{args.ligand_id}_input.txt"), "w") as f:
        f.write(f"atom_types = {atom_types}\n\n")
        f.write(f"edge_idxs = {edge_idxs}\n\n")
        f.write(f"edge_types = {edge_types}\n\n")

    with open(osp.join(out_dir, f"{args.ligand_id}_rdkit_ordered.pdb"), "w") as f:
        f.write(Chem.MolToPDBBlock(mol))

    print(f"Outputs are saved in {out_dir}!\n")

    data = {"atom_types": atom_types, "edge_idxs": edge_idxs, "edge_types": edge_types}

    with open(osp.join(out_dir, f"{args.ligand_id}.pkl"), "wb") as f:
        pkl.dump(data, f)

    with open(osp.join(out_dir, f"{args.ligand_id}.pkl"), "rb") as f:
        loaded_data = pkl.load(f)
        print(f"loaded_data:\n{loaded_data}")
