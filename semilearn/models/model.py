from torch import nn
from torchvision.models.efficientnet import (
    EfficientNet,
    _efficientnet_conf,
    EfficientNet_B0_Weights
)


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes, load_imagenet_weights=True) -> None:
        super(EfficientNetB0, self).__init__()

        inverted_residual_setting, last_channel = _efficientnet_conf(
            "efficientnet_b0", width_mult=1.0, depth_mult=1.0
        )

        self.model = EfficientNet(
            inverted_residual_setting,
            dropout=0.2,
            num_classes=num_classes,
            last_channel=last_channel,
        )

        if load_imagenet_weights:
            self.load_imagenet_weights()

    def load_imagenet_weights(self):
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        weights_dict = weights.get_state_dict(progress=True)

        # remove the classifier weight and bias
        for key in ["classifier.1.weight", "classifier.1.bias"]:
            del weights_dict[key]

        # load model weights
        self.model.load_state_dict(
            weights_dict, strict=False
        )  # strict false because the classifier layer does not match with IMAGENET

    def forward(self, x):
        return self.model(x)


# define a simple CNN model


class BasicNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.2):
        super(BasicNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from torchvision.models.efficientnet import EfficientNet_B0_Weights
    import torch

    model = EfficientNetB0(num_classes=2)
    input_batch = torch.randn(size=(2,3,224,224))
    out_logits = model(input_batch)
    print(out_logits.shape)
