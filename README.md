# AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ADicksonLab/AGDIFF/blob/main/LICENSE)

The official implementation of AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction. AGDIFF introduces a novel approach that enhances diffusion models with attention mechanisms and an improved SchNet architecture, achieving state-of-the-art performance in predicting molecular geometries.


### Unique Features of AGDIFF

- **Attention Mechanisms**: Enhances the global and local encoders with attention mechanisms for better feature extraction and integration.
- **Improved SchNet Architecture**: Incorporates learnable activation functions, adaptive scaling modules, and dual pathway processing to increase model expressiveness.
- **Batch Normalization**: Stabilizes training and improves convergence for the local encoder.
- **Feature Expansion**: Extends the MLP Edge Encoder with feature expansion and processing, combining processed features and bond embeddings for more adaptable edge representations.

For detailed information, refer to our paper on AGDIFF.

## Dataset

### Official Dataset
The official raw GEOM dataset is available [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

## Training

AGDIFF's training details and hyper-parameters are provided in the config files (`./configs/*.yml`). Feel free to tune these parameters as needed.

To train the model, use the following commands:

```bash
# Default settings
python train.py qm9_default.yml
python train.py drugs_default.yml
``` 
Model checkpoints, configuration YAML files, and training logs will be saved in a directory specified by `--logdir` in `train.py`.

## Generation

To generate conformations for entire or part of test sets, use:

```bash 
python test.py ${log}/${model}/checkpoints/${iter}.pt \
    --start_idx 800 --end_idx 1000
``` 

Here `start_idx` and `end_idx` specify the range of the test set. All sampling hyper-parameters can be set in `test.py`. For testing the QM9 model, you can add the argument `--w_global 0.3` for slightly better results.

## Evaluation

After generating conformations, evaluate the results of benchmark tasks using the following commands.

### Task 1. Conformation Generation

Calculate `COV` and `MAT` scores on the GEOM datasets with:

```bash
python eval_covmat.py ${log}/${model}/${sample}/sample_all.pkl
```

