from torch.utils.data import Dataset
from os.path import join


class CSVDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        csv_path: str,
        img_transform=None,
        label_transform=None,
        label_column_name="label",
        image_column_name="id",
        suffix=".jpg",
        start_index=0,
        end_index=-1 # only take this many samples from CSV
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.csv_path = csv_path
        self.image_column_name = image_column_name
        self.label_column_name = label_column_name
        self.suffix = suffix
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_paths, self.labels = self.read_csv(start_index,end_index)
        

    def read_csv(self,start_index,end_index):
        import pandas as pd

        df = pd.read_csv(self.csv_path)
        image_paths = df[self.image_column_name].to_list()[start_index:end_index]
        # allow empty labels for test data sets
        if "label" in df.columns:
            label_paths = df[self.label_column_name].to_list()[start_index:end_index]
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
            img = self.img_transform(img)

        if self.label_transform and label:
            label = self.label_transform(label)

        return img, label
