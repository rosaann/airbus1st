from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=W0611
import types

import torch
import torch.nn.functional as F
import torchvision.models
import pretrainedmodels


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels // 4, (1, 1)),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (3, 3), stride=stride, padding=1, output_padding=output_padding),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels // 4, out_channels, (1, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Resnet18Segmenter(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.decoder1 = Decoder(512, 256, stride=2, output_padding=1)
        self.decoder2 = Decoder(256, 128, stride=2, output_padding=1)
        self.decoder3 = Decoder(128, 64, stride=2, output_padding=1)
        self.decoder4 = Decoder(64, 64, stride=1, output_padding=0)
        self.classifier = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, num_classes, (2, 2), stride=2)
        )

    def forward(self, x):
        x = x['image']
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x = self.decoder1(x4) + x3
        x = self.decoder2(x) + x2
        x = self.decoder3(x) + x1
        x = self.decoder4(x)
        return {'mask': self.classifier(x)}
    
class Resnet18Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x['image']
        return {'has_ships': self.resnet(x)}
def get_resnet18_segmenter(num_classes=1, **kwargs):
    return Resnet18Segmenter(1)

def get_resnet18_classifier(num_classes=1, **kwargs):
    return Resnet18Classifier(1)

def get_model(model_name, params = None):
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    if params is None:
        return f()
    else:
        return f(**params)


if __name__ == '__main__':
    print('main')
    model = get_resnet34()
