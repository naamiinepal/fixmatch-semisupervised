import medmnist
from medmnist import INFO

from torchvision import transforms
IMG_SIZE = 28
# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=IMG_SIZE),
    transforms.Normalize(mean=[.5], std=[.5])
])

data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)