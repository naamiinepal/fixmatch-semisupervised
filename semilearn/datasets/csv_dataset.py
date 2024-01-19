from torch.utils.data import Dataset
from os.path import join

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets)
    if targets are not given, BasicDataset returns None as label.
    This class supports strong augmentation for FixMatch,
    and returns both weakly and strongly augmented images.
    """
    def __init__(self,image_transform,label_transform,
                 is_ulb=False,
                 strong_transform=None):
        super(BasicDataset, self).__init__()
        self.is_ulb = is_ulb
        self.image_transform = image_transform
        self.label_transform = label_transform
        if self.is_ulb:
            assert self.strong_transform is not None
        
    def __getitem__(self, index):
        """
        if strong augmentation is None,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target 
        """
        pass

class CSVDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        csv_path: str,
        img_transform=None,
        label_transform=None,
        label_column_name=None,
        image_column_name=None,
        strong_transform=None,
        is_ulb=False,
        suffix=".jpg",
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.csv_path = csv_path
        self.image_column_name = image_column_name
        self.label_column_name = label_column_name
        self.suffix = suffix
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_paths, self.labels = self.read_csv()
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform
        if self.is_ulb:
            assert self.strong_transform is not None

    def read_csv(self):
        import pandas as pd

        df = pd.read_csv(self.csv_path,header=None)
        if self.image_column_name is None:
            image_paths = df[0].to_list()
        else:
            image_paths = df[self.image_column_name].to_list()
        
        # allow empty labels for test data sets
        if self.label_column_name and (self.label_column_name in df.columns):
            label_paths = df[self.label_column_name].to_list()
        elif len(df.keys()) >=2:
            label_paths = df[1].to_list()
        else:
            label_paths = None
        return image_paths, label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = join(self.data_root_dir, self.image_paths[index])
        if self.suffix:
            img = img + self.suffix

        if self.labels:
            label = self.labels[index]
        else:
            label = None

        if self.img_transform:
            weak_augment_img = self.img_transform(img)

        if self.is_ulb:
            strong_augment_img = self.strong_transform(img)

        if self.label_transform and label is not None:
            label = self.label_transform(label)

        if self.is_ulb:
            return weak_augment_img, strong_augment_img, label
        
        return weak_augment_img, label
