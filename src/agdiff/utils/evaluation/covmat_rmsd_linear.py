import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from torch_geometric.data import Data
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from functools import partial

from ..chem import set_rdmol_positions, get_best_rmsd

def get_rmsd_confusion_matrix(data: Data, useFF=False):
    data['pos_ref'] = data['pos_ref'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)
    data['pos_gen'] = data['pos_gen'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)
    num_gen = data['pos_gen'].shape[0]
    num_ref = data['pos_ref'].shape[0]

    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen], dtype=float)
    
    for i in range(num_gen):
        gen_mol = set_rdmol_positions(data['rdmol'], data['pos_gen'][i])
        if useFF:
            MMFFOptimizeMolecule(gen_mol)
        for j in range(num_ref):
            ref_mol = set_rdmol_positions(data['rdmol'], data['pos_ref'][j])
            rmsd_confusion_mat[j, i] = get_best_rmsd(gen_mol, ref_mol)

    return rmsd_confusion_mat


def print_covmat_results(results, print_fn=print):
    df = pd.DataFrame({
        'COV-R_mean': np.mean(results.CoverageR, 0),
        'COV-R_median': np.median(results.CoverageR, 0),
        'COV-R_std': np.std(results.CoverageR, 0),
        'COV-P_mean': np.mean(results.CoverageP, 0),
        'COV-P_median': np.median(results.CoverageP, 0),
        'COV-P_std': np.std(results.CoverageP, 0),
    }, index=results.thresholds)
    print_fn('\n' + str(df))
    print_fn('MAT-R_mean: %.4f | MAT-R_median: %.4f | MAT-R_std %.4f' % (
        np.mean(results.MatchingR), np.median(results.MatchingR), np.std(results.MatchingR)
    ))
    print_fn('MAT-P_mean: %.4f | MAT-P_median: %.4f | MAT-P_std %.4f' % (
        np.mean(results.MatchingP), np.median(results.MatchingP), np.std(results.MatchingP)
    ))
    return df

class CovMatEvaluator(object):

    def __init__(self, 
        num_workers=8, 
        use_force_field=False, 
        thresholds=np.arange(0.05, 3.05, 0.05),
        ratio=2,
        filter_disconnected=True,
        print_fn=print,
        top_n=5,
        max_num_gen=5  # Added max_num_gen parameter
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.thresholds = np.array(thresholds).flatten()
        self.ratio = ratio
        self.filter_disconnected = filter_disconnected
        self.print_fn = print_fn
        self.top_n = top_n
        self.max_num_gen = max_num_gen  # Initialize max_num_gen

    def __call__(self, packed_data_list, start_idx=0):
        filtered_data_list = []
        for data in packed_data_list:
            if 'pos_gen' not in data or 'pos_ref' not in data: continue
            if self.filter_disconnected and ('.' in data['smiles']): continue
            
            data['pos_ref'] = data['pos_ref'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)
            data['pos_gen'] = data['pos_gen'].reshape(-1, data['rdmol'].GetNumAtoms(), 3)

            num_gen = data['pos_ref'].shape[0] * self.ratio
            if data['pos_gen'].shape[0] < num_gen: continue
            data['pos_gen'] = data['pos_gen'][:num_gen]

            # Limit the number of generated conformations to max_num_gen
            if data['pos_gen'].shape[0] > self.max_num_gen:
                data['pos_gen'] = data['pos_gen'][:self.max_num_gen]

            filtered_data_list.append(data)

        filtered_data_list = filtered_data_list[start_idx:]
        self.print_fn('Filtered: %d / %d' % (len(filtered_data_list), len(packed_data_list)))

        covr_scores = []
        matr_scores = []
        covp_scores = []
        matp_scores = []
        smiles_list = []
        min_rmsd_list = []
        avg_best_n_rmsd_list = []
        rmsd_confusion_mats = []
        ref_rmsd_list = []
        gen_rmsd_list = []

        for idx, data in enumerate(tqdm(filtered_data_list, total=len(filtered_data_list))):
            confusion_mat = get_rmsd_confusion_matrix(data, useFF=self.use_force_field)
            rmsd_ref_min = np.sort(confusion_mat, axis=-1) #[:, :self.top_n]
            rmsd_gen_min = np.sort(confusion_mat, axis=0) #[:self.top_n, :]
            rmsd_cov_thres = rmsd_ref_min[:, 0].reshape(-1, 1) <= self.thresholds.reshape(1, -1)
            rmsd_jnk_thres = rmsd_gen_min[0].reshape(-1, 1) <= self.thresholds.reshape(1, -1)

            num_gen = data['pos_gen'].shape[0]
            num_ref = data['pos_ref'].shape[0]
            smiles = data['smiles']

            # Calculate the average of the 5 best RMSD values for each reference molecule
            avg_best_5_rmsd = rmsd_ref_min.mean(axis=-1).mean()

            self.print_fn(f"Number of conformations for molecule {idx+1}: {num_gen}")
            self.print_fn(f"Number of references for molecule {idx+1}: {num_ref}")
            self.print_fn(f"SMILES for molecule {idx+1}: {smiles}")
            self.print_fn(f"Average of the best 5 RMSD for molecule {idx+1}: {avg_best_5_rmsd:.4f}")

            smiles_list.append(smiles)
            min_rmsd_list.append(rmsd_gen_min[0].min())
            avg_best_n_rmsd_list.append(rmsd_ref_min.mean(axis=-1).mean())
            rmsd_confusion_mats.append(confusion_mat)
            ref_rmsd_list.append(rmsd_ref_min)
            gen_rmsd_list.append(rmsd_gen_min)

            matr_scores.append(rmsd_ref_min[:, 0].mean())
            covr_scores.append(rmsd_cov_thres.mean(0, keepdims=True))
            matp_scores.append(rmsd_gen_min[0].mean())
            covp_scores.append(rmsd_jnk_thres.mean(0, keepdims=True))

        covr_scores = np.vstack(covr_scores)
        matr_scores = np.array(matr_scores)
        covp_scores = np.vstack(covp_scores)
        matp_scores = np.array(matp_scores)

        results = EasyDict({
            'CoverageR': covr_scores,
            'MatchingR': matr_scores,
            'thresholds': self.thresholds,
            'CoverageP': covp_scores,
            'MatchingP': matp_scores,
            'SMILES': smiles_list,
            'MinRMSD': min_rmsd_list,
            'AvgBestNRMSD': avg_best_n_rmsd_list,
            'RMSDConfusionMat': rmsd_confusion_mats,
            'RefRMSD': ref_rmsd_list,
            'GenRMSD': gen_rmsd_list
        })
        return results
