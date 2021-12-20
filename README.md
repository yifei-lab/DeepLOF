# DeepLOF

## Synopsis

DeepLOF is a deep learning model for predicting genes intolerant to loss-of-function (LOF) mutations. Unlike models trained on population genomic data alone, such as [LOEUF](https://www.nature.com/articles/s41586-020-2308-7), DeepLOF integrates functional and population genomic data to boost the statistical power for inferring LOF intolerance.

## Precomputed scores

Precomputed DeepLOF scores from the DeepLOF manuscript are available at [Penn State's ScholarSphere](https://scholarsphere.psu.edu/resources/946aee88-8bdc-45f6-8be7-c59f2d45b819).

## Requirements

DeepLOF is implemented in Python 3 with TensorFlow 2, TensorFlow Probability, numpy, pandas, and scikit-learn. It has been extensively tested in the following environment.

- python 3.8.10
- TensorFlow 2.4.1
- TensorFlow Probability 0.12.2
- numpy 1.19.2
- pandas 1.2.4
- scikit-learn 0.24.2

## Input file

DeepLOF requires a tab-separated text file with the following format for model training.

```
ensembl            obs_lof   exp_lof    feature_1    feature_2    ...
ENSG00000000003    3         7.865      -0.613       0.950        ...
ENSG00000000005    6         13.01      -0.068       0.704        ...
ENSG00000000419    9         17.98      0.4097       -0.11        ...
ENSG00000000457    8         34.32      -0.283       1.201        ...
ENSG00000000460    23        44.64      -0.610       0.425        ...
```

Each row of the file is a gene. The first column is the unique identifier of a gene. The second column is the observed number of LOF variants. The third column is the expected number of LOF variants under a neutral mutation model. The following columns are one or more genomic features potentially predictive of LOF intolerance. The input file used in the DeepLOF manuscript can be obtained from [Penn State's ScholarSphere](https://scholarsphere.psu.edu/resources/946aee88-8bdc-45f6-8be7-c59f2d45b819).

## Output files

DeepLOF outputs a tab-separated file with the following format. The first column is the unique identifier of a gene, and the second column is the computed DeepLOF score. A higher DeepLOF score indicates stronger negative selection against LOF mutations.

```
ensembl          DeepLOF_score
ENSG00000000003  0.431136
ENSG00000000005  0.4819911
ENSG00000000419  0.6042821
ENSG00000000457  0.6856313
ENSG00000000460  0.40926027
```

DeepLOF also outputs an hdf5 file with the state of the model from the best epoch. This file should only be used for debugging purposes.

## Running DeepLOF

```
python DeepLOF.py [OPTIONS] --input <tab-separated input file> --output <path to output files>
```

DeepLOF has two required arguments.

* `--input` Path to the input file.
The input file should be a tab-separated file as specified above.

* `--output` Path to the output files as specified above.

There are also several optional arguments.

* `--hidden` Number of hidden units in the forward neural network.
The default is no hidden unit (linear DeepLOF model).

* `--learning-rate` Learning rate of the Adam algorithm.
The default learning rate is 0.001.

* `--penalty` L2 Penalty.
The default is 0 (no L2 penalty).

* `--dropout` Dropout rate for hidden units.
The default value is 0.5.

* `--patience` Number of epochs after which training will be stopped if loss doesn't decrease in validation data.
The default is 5.

* `--epochs` Maximum number of epochs.
The default is 100.

* `--batch` Number of genes in each mini-batch.
The default is 64.

* `--validation-fraction` Fraction of genes in the validation data set.
The default is 0.2.

* `--data-seed` Random seed for splitting input data into the training and validation sets.

* `--model-seed` Random seed for initializing model parameters.

* `--save-model` Output path of the trained model.
