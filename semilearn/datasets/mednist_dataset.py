import os
from typing import Union
import numpy as np
import torch
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    RandAdjustContrast,
    MedianSmooth,
    RandCoarseShuffle,
    RandGaussianNoise,
    RandGaussianSharpen,
    ScaleIntensity,
    EnsureType,
)
import torchvision
from semilearn.datasets.csv_dataset import CSVDataset


MedNIST_LABELS = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
MedNIST_LABELS_TO_IDX = {l:idx for idx, l in enumerate(MedNIST_LABELS)}

n_classes = len(MedNIST_LABELS)
n_channels = 1

def mednist_label_to_idx(label):
    return MedNIST_LABELS_TO_IDX[label]

def convert_to_long_type(label):
    return torch.tensor(label).long()


train_transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)

strong_transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        # apply aggressive transforms
        RandAdjustContrast(1.0),
        MedianSmooth(),
        RandCoarseShuffle(holes=4,spatial_size=(6,6),max_spatial_size=(12,12),prob=0.8),
        RandGaussianNoise(prob=0.8),
        RandGaussianSharpen(prob=0.8),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)
val_transform = Compose(
    [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(), EnsureType()])


def mednist_label_transform(label:Union[str,float]):
    '''label can be either a string or float representing benign if 0.0, malignant if 1.0'''
    if isinstance(label,str):
        return mednist_label_to_idx(label)
    
    return convert_to_long_type(label)

label_transform = torchvision.transforms.Lambda(mednist_label_transform) # because test csv labels are in 0.0,1.0 format



labelled_train_csv = 'datasets/medmnist/labelled.csv'
unlabelled_train_csv = 'datasets/medmnist/unlabelled.csv'
val_csv = 'datasets/medmnist/val.csv'
test_csv = 'datasets/medmnist/test.csv'

def get_train_dataset(img_transform=train_transform,dataset_type='supervised_only'):
    if dataset_type == 'supervised_only':
        csv_path = labelled_train_csv
    elif dataset_type == 'unlabelled':
        csv_path = unlabelled_train_csv
    else:
        raise ValueError(f'dataset_type cannot be {dataset_type}')
    data_root_dir = '.'

    
    if dataset_type == 'unlabelled':
        return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,strong_transform=strong_transform,is_ulb=True)
    else:
        return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,is_ulb=False)

    




if __name__ == '__main__':
    pass
