from unittest import TestCase
from model import EfficientNetB0
from dataset import train_dataset, test_dataset

class TestSSL(TestCase):
    def test_model(self):
        model = EfficientNetB0(num_classes=7)
        print(model)

    def test_dataset(self):
        batch = next(iter(train_dataset))
        print(batch[0].shape,batch[1].shape)