from regex import D
from ssl_fork.datasets.augmentations.transforms import get_image_transform
import torchvision
from ssl_fork.datasets.csv_dataset import CSVDataset

data_root_dir = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_Data'
csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
IMG_SIZE = 224
img_transform = get_image_transform(IMG_SIZE)

ISIC_LABELS  = ['benign','malignant']
ISIC_LABELS_TO_IDX = {l:idx for idx, l in enumerate(ISIC_LABELS)}

n_classes = len(ISIC_LABELS)
n_channels = 3

def isic_label_to_idx(label):
    return ISIC_LABELS_TO_IDX[label]

label_transform = torchvision.transforms.Lambda(isic_label_to_idx)

def get_dataset(start_index,end_index,img_transform=img_transform):
    return CSVDataset(data_root_dir,csv_path,img_transform,label_transform,start_index=start_index,end_index=end_index)


if __name__ == '__main__':
    from ssl_fork.datasets.augmentations.transforms import get_image_transform,get_image_strong_augment_transform
    strong_augment_transform = get_image_strong_augment_transform(IMG_SIZE)
    image_transform = get_image_transform(IMG_SIZE)

    isic_dataset = CSVDataset(data_root_dir,csv_path,img_transform=None,label_transform=label_transform)

    img,label = isic_dataset[0]

    weak_augment = image_transform(img)
    strong_augment = strong_augment_transform(img)

    torchvision.transforms.ToPILImage()(weak_augment).show()
    torchvision.transforms.ToPILImage()(strong_augment).show()