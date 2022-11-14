import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                )

        self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),

                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),

                nn.Linear(1024, num_classes),
                nn.BatchNorm1d(num_classes),
                nn.Softmax()
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x

def vgg_cifar10(**kwargs):
    num_classes = kwargs.get('num_classes', 10)
    return CNN(num_classes)
