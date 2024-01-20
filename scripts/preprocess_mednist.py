import os
import PIL
from monai.apps import download_and_extract
import numpy as np
import pandas as pd

def download_dataset(root_dir):
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"    
    
    
    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

# download raw dataset
MEDNIST_DATA_DIR = 'datasets/medmnist'
download_dataset(MEDNIST_DATA_DIR)

# unzipped data directory
unzipped_data_dir = MEDNIST_DATA_DIR +'/MedNIST'
# obtain class labels
class_names = sorted(x for x in os.listdir(unzipped_data_dir)
                     if os.path.isdir(os.path.join(unzipped_data_dir, x)))
num_class = len(class_names)
# list class labels
print(class_names)

# gather files for each classes and key as seperate array per class label
image_files = [
    [
        os.path.join(unzipped_data_dir, class_names[i], x)
        for x in os.listdir(os.path.join(unzipped_data_dir, class_names[i]))
    ]
    for i in range(num_class)
]
label_count = [len(image_files[i]) for i in range(num_class)]

# count total
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * label_count[i])
num_total = len(image_class)

# sample image width and height
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {label_count}")


# prepare train, val, test dataset
val_frac = 0.2
test_frac = 0.2
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split

test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]

labeled_train_indices = indices[val_split:val_split +100]
unlabeled_train_indices = indices[val_split+100:]

# labelled data
labeled_train_x = [image_files_list[i] for i in labeled_train_indices]
labeled_train_y = [image_class[i] for i in labeled_train_indices]

labeled_train_dict = {0:labeled_train_x, 1:labeled_train_y}
pd.DataFrame.from_dict(labeled_train_dict).to_csv(f'{MEDNIST_DATA_DIR}/labelled.csv',header=None,index=False)

#unlabelled data, but we keep the labels too, for debugging
unlabeled_train_x = [image_files_list[i] for i in unlabeled_train_indices]
unlabeled_train_y = [image_class[i] for i in unlabeled_train_indices]

unlabeled_train_dict = {0:unlabeled_train_x, 1:unlabeled_train_y}
pd.DataFrame.from_dict(unlabeled_train_dict).to_csv(f'{MEDNIST_DATA_DIR}/unlabelled.csv',header=None,index=False)

# validation set
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]


val_dict = {0:val_x, 1:val_y}
pd.DataFrame.from_dict(val_dict).to_csv(f'{MEDNIST_DATA_DIR}/val.csv',header=None,index=False)

# test set
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

test_dict = {}
pd.DataFrame.from_dict(val_dict).to_csv(f'{MEDNIST_DATA_DIR}/test.csv',header=None,index=False)

print(
    f"Labeled Training count: {len(labeled_train_x)},\nUnlabeled Training count: {len(unlabeled_train_x)},\nValidation count: "
    f"{len(val_x)}, \nTest count: {len(test_x)}")