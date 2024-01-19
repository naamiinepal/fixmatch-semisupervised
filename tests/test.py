from unittest import TestCase

import torch
from semilearn.models.model import EfficientNetB0
from semilearn.datasets.medmnist_dataset import train_dataset, test_dataset
from semilearn.datasets.csv_dataset import CSVDataset
from semilearn.core.criterions.cross_entropy import ce_loss

class TestSSL(TestCase):
    def test_model(self):
        model = EfficientNetB0(num_classes=7)
        # print(model)

    def test_dataset(self):
        batch = next(iter(train_dataset))
        print(batch[0].shape,batch[1].shape)

    def test_csv_dataset(self):
        data_root_dir = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_Data'
        csv_path = 'isic_challenge/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
        isic_dataset = CSVDataset(data_root_dir=data_root_dir,csv_path=csv_path)
        print(len(isic_dataset))
        # [print(isic_dataset[0])]
    
    def test_ce_loss(self):
        logits_pred = torch.randn(1,5)
        idx = torch.argmax(logits_pred)
        target = torch.zeros_like(logits_pred)
        target.index_fill_(dim=-1,index=idx,value=1.0)
        print(logits_pred,target)
        loss_val = ce_loss(logits_pred,target)
        print(loss_val)