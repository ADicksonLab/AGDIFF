import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PyPDF2 import PdfFileReader, PdfFileWriter

from utils.datasets import PackedConformationDataset
#from utils.evaluation.covmat_rmsd import CovMatEvaluator, print_covmat_results
from utils.evaluation.covmat_rmsd_linear import CovMatEvaluator, print_covmat_results
from utils.misc import *
import matplotlib.cm as cm
from collections import defaultdict

import matplotlib
import matplotlib.colors as mcolors
from matplotlib import rc

matplotlib.use('Agg')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Helper functions
def tokenize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    tokens = set()
    for atom in mol.GetAtoms():
        frag = Chem.MolFragmentToSmiles(mol, [atom.GetIdx()])
        frag_mol = Chem.MolFromSmiles(frag)
        if frag_mol is not None:
            token = Chem.MolToSmiles(frag_mol)
            tokens.add(token)
    return tokens

def extract_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_with_h = Chem.AddHs(mol)  # Add implicit hydrogens to the molecule
    num_atoms = mol_with_h.GetNumAtoms()  # Count all atoms, including implicit hydrogens
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return {
        'Number of Atoms': num_atoms,
        'Topological Polar Surface Area': tpsa,
        'LogP': logp,
        'Number of Hydrogen Donors': num_h_donors,
        'Number of Hydrogen Acceptors': num_h_acceptors,
        'Number of Rotatable Bonds': num_rotatable_bonds,
        'Fingerprint': fingerprint,
        'Scaffold': scaffold
    }

def sanitize_smiles(smiles):
    return smiles.replace('#', '\#')

def auto_crop_pdf(input_path, output_path):
    input_pdf = PdfFileReader(input_path)
    output_pdf = PdfFileWriter()
    num_pages = input_pdf.getNumPages()
    for i in range(num_pages):
        page = input_pdf.getPage(i)
        page.cropBox.lowerLeft = (10, 10)
        page.cropBox.upperRight = (page.mediaBox.getUpperRight_x() - 10, page.mediaBox.getUpperRight_y() - 10)
        output_pdf.addPage(page)
    with open(output_path, 'wb') as f:
        output_pdf.write(f)
    os.remove(input_path)  # Remove the original file

def plot_analysis(smiles, avg_best_n_rmsd, rmsd_confusion_mat, ref_rmsd, gen_rmsd, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    features = [extract_molecular_features(s) for s in smiles]
    df = pd.DataFrame(features)
    df['SMILES'] = smiles
    df['Sanitized SMILES'] = df['SMILES'].apply(sanitize_smiles)
    df['Average of Best 5 RMSD (Å)'] = avg_best_n_rmsd

    df.dropna(subset=['Number of Atoms'], inplace=True)
    df['Number of Atoms'] = df['Number of Atoms'].astype(int)

    feature_columns = ['Number of Atoms', 'Topological Polar Surface Area', 'LogP', 'Number of Hydrogen Donors', 'Number of Hydrogen Acceptors', 'Number of Rotatable Bonds']
    X = df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=5, random_state=0).fit(X_pca)
    df['Cluster'] = kmeans.labels_

    # Plotting improvements
    sns.set_context("notebook", font_scale=1.4)
    sns.set_style("whitegrid")

    # Cumulative Distribution of RMSD Values
    sorted_ref_rmsd = np.sort(ref_rmsd)
    sorted_gen_rmsd = np.sort(gen_rmsd)
    cdf_ref = np.arange(len(sorted_ref_rmsd)) / float(len(sorted_ref_rmsd))
    cdf_gen = np.arange(len(sorted_gen_rmsd)) / float(len(sorted_gen_rmsd))
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_ref_rmsd, cdf_ref, label='Reference RMSD', color='blue')
    plt.plot(sorted_gen_rmsd, cdf_gen, label='Generated RMSD', color='red')
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Cumulative Frequency')
    plt.title('Cumulative Distribution of RMSD Values')
    plt.xlim(0, 4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_rmsd_distribution.pdf'))
    plt.savefig(os.path.join(output_dir, 'cumulative_rmsd_distribution.png'))

    plt.close()

    auto_crop_pdf(os.path.join(output_dir, 'cumulative_rmsd_distribution.pdf'), os.path.join(output_dir, 'cumulative_rmsd_distribution_cropped.pdf'))

    # Scatter Plot: Generated vs Reference RMSD
    plt.figure(figsize=(12, 8))
    plt.scatter(gen_rmsd, ref_rmsd, label='RMSD Values', color='blue', edgecolor='black')
    plt.plot([0, max(gen_rmsd)], [0, max(ref_rmsd)], 'k--', lw=2, label='y=x', color='red')
    plt.title('Generated RMSD vs. Reference RMSD')
    plt.xlabel('Generated RMSD (Å)')
    plt.ylabel('Reference RMSD (Å)')
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gen_vs_ref_rmsd.pdf'))
    plt.savefig(os.path.join(output_dir, 'gen_vs_ref_rmsd.png'))

    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'gen_vs_ref_rmsd.pdf'), os.path.join(output_dir, 'gen_vs_ref_rmsd_cropped.pdf'))

    # Distribution of Average of Best 5 RMSD
    plt.figure(figsize=(12, 8))
    cmap = mcolors.LinearSegmentedColormap.from_list("rmsd_cmap", ["#440154", "#20908D", "#FDE724"])
    norm = plt.Normalize(vmin=df['Average of Best 5 RMSD (Å)'].min(), vmax=df['Average of Best 5 RMSD (Å)'].max())
    n, bins, patches = plt.hist(df['Average of Best 5 RMSD (Å)'], bins=30, alpha=0.7, edgecolor='black')
    for i in range(len(patches)):
        patches[i].set_facecolor(cmap(norm((bins[i] + bins[i + 1]) / 2)))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Average of Best 5 RMSD (Å)')
    plt.title('Distribution of Average of Best 5 RMSD')
    plt.xlabel('Average of Best 5 RMSD (Å)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'rmsd_distribution.pdf'))
    plt.savefig(os.path.join(output_dir, 'rmsd_distribution.png'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'rmsd_distribution.pdf'), os.path.join(output_dir, 'rmsd_distribution_cropped.pdf'))

    # Correlation Matrix
    plt.figure(figsize=(14, 12))
    corr_matrix = df[feature_columns + ['Average of Best 5 RMSD (Å)']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12}, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Molecular Properties and RMSD')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.pdf'))
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'correlation_matrix.pdf'), os.path.join(output_dir, 'correlation_matrix_cropped.pdf'))

    # Correlation Matrix (only last row)
    plt.figure(figsize=(16, 4))  # Adjust the height to focus on the last row
    corr_matrix = df[feature_columns + ['Average of Best 5 RMSD (Å)']].corr()
    corr_last_row = corr_matrix[['Average of Best 5 RMSD (Å)']].T  # Transpose to make it a row
    sns.heatmap(corr_last_row, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12}, vmin=-1, vmax=1, cbar=False)
    plt.title('Correlation of Molecular Properties with Average of Best 5 RMSD')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_last_row.pdf'))
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_last_row.png'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'correlation_matrix_last_row.pdf'), os.path.join(output_dir, 'correlation_matrix_last_row_cropped.pdf'))


    '''# Average RMSD Confusion Matrix
    avg_rmsd_matrix = np.mean(np.array(rmsd_confusion_mat), axis=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_rmsd_matrix, cmap='viridis')
    plt.title('Average RMSD Confusion Matrix')
    plt.xlabel('Generated Conformations')
    plt.ylabel('Reference Conformations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_rmsd_confusion_matrix.pdf'))
    plt.savefig(os.path.join(output_dir, 'avg_rmsd_confusion_matrix.png'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'avg_rmsd_confusion_matrix.pdf'), os.path.join(output_dir, 'avg_rmsd_confusion_matrix_cropped.pdf'))
    '''
    '''# Scatter Plot: Generated vs Reference RMSD
    plt.figure(figsize=(12, 8))
    plt.scatter(ref_rmsd, gen_rmsd, color='blue', edgecolor='black', alpha=0.7, s=100)
    plt.plot([0, max(ref_rmsd.max(), gen_rmsd.max())], [0, max(ref_rmsd.max(), gen_rmsd.max())], 'k--', lw=2)
    plt.title('Generated RMSD vs. Reference RMSD')
    plt.xlabel('Reference RMSD (Å)')
    plt.ylabel('Generated RMSD (Å)')
    plt.xlim(0, max(ref_rmsd.max(), gen_rmsd.max()) + 0.5)
    plt.ylim(0, max(ref_rmsd.max(), gen_rmsd.max()) + 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gen_vs_ref_rmsd.pdf'))
    plt.savefig(os.path.join(output_dir, 'gen_vs_ref_rmsd.png'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'gen_vs_ref_rmsd.pdf'), os.path.join(output_dir, 'gen_vs_ref_rmsd.pdf'))
    '''
    # PCA of Molecular Descriptors
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis', s=100, edgecolor='black')
    plt.title('PCA of Molecular Descriptors')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(output_dir, 'pca_clusters.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'pca_clusters.pdf'), os.path.join(output_dir, 'pca_clusters_cropped.pdf'))

    # Violin Plot of RMSD by Cluster and LogP
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Cluster', y='Average of Best 5 RMSD (Å)', data=df, palette='viridis', inner=None)
    sns.swarmplot(x='Cluster', y='Average of Best 5 RMSD (Å)', hue='LogP', data=df, palette='coolwarm', dodge=True, size=3)
    plt.title('Violin Plot of RMSD by Cluster and LogP')
    plt.xlabel('Cluster')
    plt.ylabel('Average of Best 5 RMSD (Å)')
    plt.legend(title='LogP', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmsd_by_cluster_logp.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'rmsd_by_cluster_logp.pdf'), os.path.join(output_dir, 'rmsd_by_cluster_logp_cropped.pdf'))

    # Distribution of Number of Atoms
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Number of Atoms', kde=True, color='darkcyan')
    plt.title('Distribution of Number of Atoms')
    plt.xlabel('Number of Atoms')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_num_atoms.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'distribution_num_atoms.pdf'), os.path.join(output_dir, 'distribution_num_atoms_cropped.pdf'))

    # Distribution of Topological Polar Surface Area
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Topological Polar Surface Area', kde=True, color='darkblue')
    plt.title('Distribution of Topological Polar Surface Area')
    plt.xlabel('Topological Polar Surface Area')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_tpsa.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'distribution_tpsa.pdf'), os.path.join(output_dir, 'distribution_tpsa_cropped.pdf'))

    # Distribution of LogP
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='LogP', kde=True, color='darkgreen')
    plt.title('Distribution of LogP')
    plt.xlabel('LogP')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_logp.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'distribution_logp.pdf'), os.path.join(output_dir, 'distribution_logp_cropped.pdf'))

    # Distribution of Number of Rotatable Bonds
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Number of Rotatable Bonds', kde=True, color='darkred')
    plt.title('Distribution of Number of Rotatable Bonds')
    plt.xlabel('Number of Rotatable Bonds')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_num_rotatable_bonds.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'distribution_num_rotatable_bonds.pdf'), os.path.join(output_dir, 'distribution_num_rotatable_bonds_cropped.pdf'))

    # Distribution of Average of Best 5 RMSD
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Average of Best 5 RMSD (Å)', kde=True, color='darkmagenta')
    plt.title('Distribution of Average of Best 5 RMSD')
    plt.xlabel('Average of Best 5 RMSD (Å)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_avg_best_5_rmsd.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'distribution_avg_best_5_rmsd.pdf'), os.path.join(output_dir, 'distribution_avg_best_5_rmsd_cropped.pdf'))

    # Scatter Plots of Molecular Properties vs. Average of Best 5 RMSD
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['Number of Atoms'], y=df['Average of Best 5 RMSD (Å)'], color='teal', edgecolor='black')
    plt.title('Number of Atoms vs. Average of Best 5 RMSD')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Average of Best 5 RMSD (Å)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'num_atoms_vs_avg_best_5_rmsd.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'num_atoms_vs_avg_best_5_rmsd.pdf'), os.path.join(output_dir, 'num_atoms_vs_avg_best_5_rmsd_cropped.pdf'))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['Topological Polar Surface Area'], y=df['Average of Best 5 RMSD (Å)'], color='teal', edgecolor='black')
    plt.title('Topological Polar Surface Area vs. Average of Best 5 RMSD')
    plt.xlabel('Topological Polar Surface Area')
    plt.ylabel('Average of Best 5 RMSD (Å)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tpsa_vs_avg_best_5_rmsd.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'tpsa_vs_avg_best_5_rmsd.pdf'), os.path.join(output_dir, 'tpsa_vs_avg_best_5_rmsd_cropped.pdf'))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['LogP'], y=df['Average of Best 5 RMSD (Å)'], color='teal', edgecolor='black')
    plt.title('LogP vs. Average of Best 5 RMSD')
    plt.xlabel('LogP')
    plt.ylabel('Average of Best 5 RMSD (Å)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logp_vs_avg_best_5_rmsd.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'logp_vs_avg_best_5_rmsd.pdf'), os.path.join(output_dir, 'logp_vs_avg_best_5_rmsd_cropped.pdf'))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['Number of Rotatable Bonds'], y=df['Average of Best 5 RMSD (Å)'], color='teal', edgecolor='black')
    plt.title('Number of Rotatable Bonds vs. Average of Best 5 RMSD')
    plt.xlabel('Number of Rotatable Bonds')
    plt.ylabel('Average of Best 5 RMSD (Å)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rotatable_bonds_vs_avg_best_5_rmsd.pdf'))
    plt.close()
    auto_crop_pdf(os.path.join(output_dir, 'rotatable_bonds_vs_avg_best_5_rmsd.pdf'), os.path.join(output_dir, 'rotatable_bonds_vs_avg_best_5_rmsd_cropped.pdf'))

    # Statistics and Correlations
    rmsd_stats = df['Average of Best 5 RMSD (Å)'].describe()
    print("RMSD Statistics:")
    print(rmsd_stats)

    print("Correlation between molecular properties and RMSD:")
    print(corr_matrix)

    df.to_csv(os.path.join(output_dir, 'analysis_results.csv'), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--ratio', type=int, default=2)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./plots')
    args = parser.parse_args()
    assert os.path.isfile(args.path)

    with open(args.path, 'rb') as f:
        packed_dataset = pickle.load(f)

    evaluator = CovMatEvaluator(ratio=args.ratio)
    results = evaluator(packed_data_list=list(packed_dataset), start_idx=args.start_idx)

    plot_analysis(results.SMILES, results.MinRMSD, results.RMSDConfusionMat, results.MatchingR, results.MatchingP, args.output_dir)
