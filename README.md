# AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ADicksonLab/AGDIFF/blob/main/LICENSE)

The official implementation of AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction.

## Dataset

### Official Dataset
The official raw GEOM dataset is available [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

### Preprocessed Dataset
We provide the preprocessed datasets (GEOM) in this [[Google Drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). After downloading the dataset, it should be put into the folder path as specified in the `dataset` variable of config files `./configs/*.yml`.

### Prepare Your Own GEOM Dataset from Scratch (Optional)

You can also download the original GEOM full dataset and prepare your own data split. A guide is available at ConfGF's [[GitHub page]](https://github.com/DeepGraphLearning/ConfGF#prepare-your-own-geom-dataset-from-scratch-optional).

## Training

All hyper-parameters and training details are provided in the config files (`./configs/*.yml`), and feel free to tune these parameters.

You can train the model with the following commands:

```bash
# Default settings
python train.py ./config/qm9_default.yml
python train.py ./config/drugs_default.yml
# An ablation setting with fewer timesteps, as described in Appendix D.2.
python train.py ./config/drugs_1k_default.yml
```

The model checkpoints, configuration YAML file, and training log will be saved into a directory specified by `--logdir` in `train.py`.

## Generation

We provide the checkpoints of two trained models, i.e., `qm9_default` and `drugs_default` in the [[Google Drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). Note that, please put the checkpoints `*.pt` into paths like `${log}/${model}/checkpoints/`, and also put the corresponding configuration file `*.yml` into the upper-level directory `${log}/${model}/`.

<font color="red">Attention</font>: if you want to use pretrained models, please use the code at the [`pretrain`](https://github.com/ADicksonLab/AGDIFF/tree/pretrain) branch, which is the vanilla codebase for reproducing the results with our pretrained models. We recently noticed some issues with the codebase and updated it, making the `main` branch not compatible well with the previous checkpoints.

You can generate conformations for entire or part of test sets by:

```bash
python test.py ${log}/${model}/checkpoints/${iter}.pt \
    --start_idx 0 --end_idx 200
```
Here `start_idx` and `end_idx` indicate the range of the test set that we want to use.  To reproduce the paper's results, you should use indexes 0 and 200. All hyper-parameters related to sampling can be set in `test.py` files. Specifically, for testing the qm9 model, you could add the additional arg `--w_global 0.3`, which empirically shows slightly better results.


## Evaluation

After generating conformations following the above commands, the results of all benchmark tasks can be calculated based on the generated data.

### Task 1. Conformation Generation

The `COV` and `MAT` scores on the GEOM datasets can be calculated using the following commands:

```bash
python eval_covmat.py ${log}/${model}/${sample}/sample_all.pkl
```
