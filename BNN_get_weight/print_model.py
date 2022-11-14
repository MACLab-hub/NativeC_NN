import torch
import torchvision
import torch.onnx
import models
import onnx
import numpy as np
import hiddenlayer as hl
from torch import nn
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from torchsummary import summary as summary

model = models.vgg_cifar10
model_config = {'input_size': None, 'dataset': 'cifar10'}
model = model(**model_config)

#model.to('cuda')
#summary(model, (1, 32, 32))
'''def weight_init(submodule):
    if isinstance(submodule, torch.nn.Conv2d) or isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d) or isinstance(submodule, torch.nn.BatchNorm1d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()
#        submodule.running_mean.data.zero_()
#        submodule.running_var.data.zero_()
model.apply(weight_init)
'''
model.eval()

# netron (https://netron.app/)
dummy_data = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
#torch.onnx.export(model, dummy_data, "output.onnx",
#                  input_names=['input'], output_names=['output'],
#                  training=torch.onnx.TrainingMode.PRESERVE)

# hiddenlayer
transforms = [hl.transforms.Prune('Constant')]
graph = hl.build_graph(model, dummy_data, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('cnn_hl', format='png')
