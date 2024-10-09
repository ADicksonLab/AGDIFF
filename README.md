# AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ADicksonLab/AGDIFF/blob/main/LICENSE)

**[paper](https://chemrxiv.org/engage/chemrxiv/article-details/6703fbffcec5d6c14273d4ce)**


This repository contains the official implementation of the work "AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction".

AGDIFF introduces a novel approach that enhances diffusion models with attention mechanisms and an improved SchNet architecture, achieving state-of-the-art performance in predicting molecular geometries.

### Unique Features of AGDIFF

- **Attention Mechanisms**: Enhances the global and local encoders with attention mechanisms for better feature extraction and integration.
- **Improved SchNet Architecture**: Incorporates learnable activation functions, adaptive scaling modules, and dual pathway processing to increase model expressiveness.
- **Batch Normalization**: Stabilizes training and improves convergence for the local encoder.
- **Feature Expansion**: Extends the MLP Edge Encoder with feature expansion and processing, combining processed features and bond embeddings for more adaptable edge representations.

<p align="center">
	<img src="assets/agdiff_framework.png" alt="photo not available" width="80%" height="80%">
</p>

 

https://github.com/user-attachments/assets/f0c636f8-4677-41c7-8f0a-009643c999d2


## Content ##
0. [Environment Setup](#environment-setup)
0. [Unique Features of AGDIFF](#unique-features-of-agdiff)
0. [Dataset](#dataset)
0. [Training](#training)
0. [Generation](#generation)
0. [Evaluation](#evaluation)
0. [Acknowledgment](#acknowledgement)
0. [Citattion](#citation)



## Environment Setup ##


### Install via Conda

```bash
conda env create -f agdiff.yml
conda activate agdiff
```



## Dataset ##

### Official Dataset
The official raw GEOM dataset is available [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

## Training ##

AGDIFF's training details and hyper-parameters are provided in the config files (`./configs/*.yml`). Feel free to tune these parameters as needed.

To train the model, use the following commands:

```bash
python train.py qm9_default.yml
python train.py drugs_default.yml
``` 
Model checkpoints, configuration YAML files, and training logs will be saved in a directory specified by `--logdir` in `train.py`.

## Generation ##

To generate conformations for entire or part of test sets, use:

```bash 
python test.py ${log}/${model}/checkpoints/${iter}.pt ./configs/qm9_default.yml \
    --start_idx 0 --end_idx 200
```
Here `start_idx` and `end_idx` indicate the range of the test set that we want to use. To reproduce the paper's results, you should use 0 and 200 for start_idx and end_idx, respectively. All hyper-parameters related to sampling can be set in `test.py` files. Specifically, for testing the qm9 model, you could add the additional arg `--w_global 0.3`, which empirically shows slightly better results.


## Evaluation ##

After generating conformations, evaluate the results of benchmark tasks using the following commands.

### Task 1. Conformation Generation

Calculate `COV` and `MAT` scores on the GEOM datasets with:

```bash
python eval_covmat.py ${log}/${model}/${sample}/sample_all.pkl
```

## Acknowledgement ##

Our implementation is based on [GEODIFF](https://github.com/MinkaiXu/GeoDiff), [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [SchNet](https://github.com/atomistic-machine-learning/SchNet)


## Citation ##

If you use our code or method in your work, please consider citing the following:

```bibtex
@misc{wyzykowskiAGDIFFAttentionEnhancedDiffusion2024,
  title = {{{AGDIFF}}: {{Attention-Enhanced Diffusion}} for {{Molecular Geometry Prediction}}},
  shorttitle = {{{AGDIFF}}},
  author = {Wyzykowski, Andr{\'e} Brasil Vieira and Niazi, Fatemeh Fathi and Dickson, Alex},
  year = {2024},
  month = oct,
  publisher = {ChemRxiv},
  doi = {10.26434/chemrxiv-2024-wrvr4},
  urldate = {2024-10-09},
  archiveprefix = {ChemRxiv},
  langid = {english},
  keywords = {attention,conformer,diffusion models,generative,GNN,graph neural network,machine learning,structure}
}
```

Please direct any questions to Alex Dickson (alexrd@msu.edu).