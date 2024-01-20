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
    Lambda,
    Resize,
)
import torchvision
from semilearn.datasets.csv_dataset import CSVDataset


MedNIST_LABELS = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]
MedNIST_LABELS_TO_IDX = {l: idx for idx, l in enumerate(MedNIST_LABELS)}

n_classes = len(MedNIST_LABELS)
n_channels = 3
IMG_SIZE = 224


def mednist_label_to_idx(label):
    return MedNIST_LABELS_TO_IDX[label]


def convert_to_long_type(label):
    return torch.tensor(label).long()


def repeat_channel(img: torch.Tensor, n_channel=3):
    return img.repeat(n_channel, 1, 1)


train_transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
        Resize(spatial_size=(IMG_SIZE, IMG_SIZE)),
        Lambda(
            repeat_channel
        ),  # to be able to use the imagenet-pretrained model which take in RGB images
    ]
)

strong_transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        # apply aggressive transforms
        RandAdjustContrast(1.0),
        MedianSmooth(),
        RandCoarseShuffle(
            holes=4, spatial_size=(6, 6), max_spatial_size=(12, 12), prob=0.8
        ),
        RandGaussianNoise(prob=0.8),
        RandGaussianSharpen(prob=0.8),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        Resize(spatial_size=(IMG_SIZE, IMG_SIZE)),
        EnsureType(),
        Lambda(repeat_channel),
    ]
)
val_transform = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size=(IMG_SIZE, IMG_SIZE)),
        EnsureType(),
        Lambda(repeat_channel),

    ]
)


def mednist_label_transform(label: Union[str, float]):
    """label can be either a string or float representing benign if 0.0, malignant if 1.0"""

    return convert_to_long_type(int(label))


label_transform = torchvision.transforms.Lambda(
    mednist_label_transform
)  # because test csv labels are in 0.0,1.0 format


labelled_train_csv = "datasets/medmnist/labelled.csv"
unlabelled_train_csv = "datasets/medmnist/unlabelled.csv"
val_csv = "datasets/medmnist/val.csv"
test_csv = "datasets/medmnist/test.csv"


def get_train_dataset(img_transform=train_transform, dataset_type="supervised_only"):
    if dataset_type == "supervised_only":
        csv_path = labelled_train_csv
    elif dataset_type == "unlabelled":
        csv_path = unlabelled_train_csv
    else:
        raise ValueError(f"dataset_type cannot be {dataset_type}")
    data_root_dir = "."

    if dataset_type == "unlabelled":
        return CSVDataset(
            data_root_dir,
            csv_path,
            img_transform,
            label_transform,
            strong_transform=strong_transform,
            is_ulb=True,suffix=None
        )
    else:
        return CSVDataset(
            data_root_dir, csv_path, img_transform, label_transform, is_ulb=False, suffix=None
        )

def get_test_dataset(img_transform=val_transform):
    csv_path = test_csv
    data_root_dir = '.'

    return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,is_ulb=False,suffix=None)

def get_val_dataset(img_transform=val_transform):
    csv_path = val_csv
    data_root_dir = '.'

    return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,is_ulb=False,suffix=None)


if __name__ == "__main__":
    sample_data_path = "datasets/medmnist/MedNIST/AbdomenCT/000000.jpeg"
    sample_img = train_transform(sample_data_path)
    print(sample_img.shape)

    dataset = get_train_dataset()
    for batch in dataset:
        img,label = batch
        print(img.shape,label)

    dataset = get_train_dataset(dataset_type='unlabelled')

    for batch in dataset:
        img,strong_img, label = batch
        print(img.shape,label)

    dataset = get_val_dataset()
    for batch in dataset:
        img,label = batch
        print(img.shape,label)

    dataset = get_test_dataset()
    for batch in dataset:
        img,label = batch
        print(img.shape,label)