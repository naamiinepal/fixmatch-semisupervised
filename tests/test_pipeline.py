import torch

from semilearn.datasets.isic_dataset import n_classes

from torchvision.models.efficientnet import EfficientNet,_efficientnet_conf

inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

model = EfficientNet(inverted_residual_setting,dropout=0.2,num_classes=n_classes,last_channel=last_channel)

rand_input = torch.rand(1,3,224,224)
pred_logits = model(rand_input)

print(rand_input.shape,pred_logits.shape)