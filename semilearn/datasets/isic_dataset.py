from typing import Union
import torch
from semilearn.datasets.augmentations.transforms import get_image_strong_augment_nocolor_transform, get_image_transform, get_image_strong_augment_transform
import torchvision
from semilearn.datasets.csv_dataset import CSVDataset

train_data_root_dir = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_Data'
fully_supervised_train_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
supervised_only_train_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth_labelled.csv'
unlabelled_train_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth_unlabelled.csv'
test_data_root_dir = 'isic_challenge/ISBI2016_ISIC_Part3B_Test_Data'
val_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Test_GroundTruth_val.csv'
test_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Test_GroundTruth_test.csv'

IMG_SIZE = 224
img_transform = get_image_transform(IMG_SIZE)
strong_transform = get_image_strong_augment_nocolor_transform(IMG_SIZE)

ISIC_LABELS  = ['benign','malignant']
ISIC_LABELS_TO_IDX = {l:idx for idx, l in enumerate(ISIC_LABELS)}

n_classes = len(ISIC_LABELS)
n_channels = 3



def isic_label_to_idx(label):
    return ISIC_LABELS_TO_IDX[label]

def convert_to_long_type(label):
    return torch.tensor(label).long()

def isic_label_transform(label:Union[str,float]):
    '''label can be either a string or float representing benign if 0.0, malignant if 1.0'''
    if isinstance(label,str):
        return isic_label_to_idx(label)
    
    return convert_to_long_type(label)

label_transform = torchvision.transforms.Lambda(isic_label_transform) # because test csv labels are in 0.0,1.0 format

def get_test_dataset(img_transform=img_transform):
    csv_path = test_csv_path
    data_root_dir = test_data_root_dir

    return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,is_ulb=False)

def get_val_dataset(img_transform=img_transform):
    csv_path = val_csv_path
    data_root_dir = test_data_root_dir

    return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,is_ulb=False)

def get_train_dataset(img_transform=img_transform,dataset_type='fully_supervised'):
    if dataset_type == 'fully_supervised':
        csv_path = fully_supervised_train_csv_path
    elif dataset_type == 'supervised_only':
        csv_path = supervised_only_train_csv_path
    elif dataset_type == 'unlabelled':
        csv_path = unlabelled_train_csv_path
    else:
        raise ValueError(f'dataset_type cannot be {dataset_type}')
    data_root_dir = train_data_root_dir

    
    if dataset_type == 'unlabelled':
        return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,strong_transform=strong_transform,is_ulb=True)
    else:
        return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,is_ulb=False)

    
if __name__ == '__main__':
    from semilearn.datasets.augmentations.transforms import get_image_transform
    from semilearn.datasets.isic_dataset import get_train_dataset,get_val_dataset,get_test_dataset
    image_transform = get_image_transform(IMG_SIZE)

    train_full_supervised = get_train_dataset(dataset_type='fully_supervised')
    train_supervised_only = get_train_dataset(dataset_type='supervised_only')
    train_unlabelled = get_train_dataset(dataset_type='unlabelled')
    val_isic_dataset = get_val_dataset()
    test_isic_dataset = get_test_dataset()

    print(f'train fully-supervised {len(train_full_supervised)} supervised-only {len(train_supervised_only)} unlabelled {len(train_unlabelled)} val {len(val_isic_dataset)} test {len(test_isic_dataset)}')


