This codebase is adapted from https://github.com/microsoft/Semi-supervised-learning

## Getting Started

To get a local copy up, running follow these simple example steps.

### Prerequisites

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
conda activate usb
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Download and Preprocess data

This code base is adapted on Two Dataset - MedNIST, Skin Lesion Dataset

```shell
sh scripts/preprocess_isic.sh
```


To start training on small labelled dataset
```sh
python train_supervised_mednist.py --supervised_only --num_epochs 30
```
To start training on small labelled  + unlabelled dataset
```sh
python train_semi-supervised_skin_cancer.py --num_epochs 30
``` 

To run tensorboard visualization
```sh
tensorboard --logdir tb-logger
```

