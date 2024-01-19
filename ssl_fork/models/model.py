from torch import nn
from torchvision.models.efficientnet import EfficientNet,_efficientnet_conf


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes) -> None:
        super(EfficientNetB0,self).__init__()

        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

        self.model = EfficientNet(inverted_residual_setting,dropout=0.2,num_classes=num_classes,last_channel=last_channel)
    
    def forward(self,x):
        return self.model(x)
    
# define a simple CNN model

class BasicNet(nn.Module):
    def __init__(self, in_channels, num_classes,dropout=0.2):
        super(BasicNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
