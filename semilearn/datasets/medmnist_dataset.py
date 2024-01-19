import medmnist
from medmnist import INFO

from torchvision import transforms
IMG_SIZE = 28

# preprocessing
pathmnist_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
    transforms.Normalize(mean=[.5], std=[.5])
])

pathmnist_weak_augment_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
    transforms.Normalize(mean=[.5],std=[.5])
])

pathmnist_strong_augment_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
    transforms.Normalize(mean=[.5],std=[.5]),
])

data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# load the data
train_dataset = DataClass(split='train', transform=pathmnist_data_transform, download=download)
train_dataset_no_transform = DataClass(split='train',transform=None,download=download)
test_dataset = DataClass(split='test', transform=pathmnist_data_transform, download=download)

if __name__ == '__main__':
    img, label = train_dataset_no_transform[0]
    print(img, label)