from typing import Union
import torch
from semilearn.datasets.augmentations.transforms import get_image_transform
import torchvision
from semilearn.datasets.csv_dataset import CSVDataset

train_data_root_dir = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_Data'
train_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
test_data_root_dir = 'isic_challenge/ISBI2016_ISIC_Part3B_Test_Data'
test_csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv'

IMG_SIZE = 224
img_transform = get_image_transform(IMG_SIZE)

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

def get_dataset(start_index=0,end_index=-1,img_transform=img_transform,train=True):
    if train:
        csv_path = train_csv_path
        data_root_dir = train_data_root_dir
    else:
        csv_path = test_csv_path
        data_root_dir = test_data_root_dir
    
    label_transform = torchvision.transforms.Lambda(isic_label_transform) # because test csv labels are in 0.0,1.0 format
    
    return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,start_index=start_index,end_index=end_index)

    
if __name__ == '__main__':
    from semilearn.datasets.augmentations.transforms import get_image_transform
    from semilearn.datasets.isic_dataset import get_dataset
    image_transform = get_image_transform(IMG_SIZE)

    train_isic_dataset = get_dataset(img_transform=img_transform,train=True)
    test_isic_dataset = get_dataset(img_transform=img_transform,train=False)

    print(f'train {len(train_isic_dataset)} test {len(test_isic_dataset)}')

    img, label = test_isic_dataset[0]
    print(f'Test sample {img.shape} {label}')

