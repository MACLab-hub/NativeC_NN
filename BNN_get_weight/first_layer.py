import torch
import models
import numpy as np
from torch import nn
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from torchsummary import summary as summary

transform = get_transform('cifar10', 32, augment=False)
val_data = get_dataset('cifar10', 'val', transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)

model = models.vgg_cifar10_binary
model_config = {'input_size': None, 'dataset': 'cifar10'}
model = model(**model_config)
print("created model with configuration: %s"%(model_config))

checkpoint = torch.load("./results/vgg_cifar10_binary/checkpoint.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
print("loaded checkpoint (epoch %s)"%(checkpoint['epoch']))

model.train(False)
with torch.no_grad():
    for i , (inputs, target) in enumerate(val_loader):
        #if i==0:
        #    continue;
        if i==1: #2:
            break;
        input_var = Variable(inputs, volatile=not True)
        target_var = Variable(target)
        
        conv0 = model.features[0](input_var)
        bn0 = model.features[1](conv0)
        tanh0 = model.features[2](bn0)
        
        conv1 = model.features[3](tanh0)
        max0 = model.features[4](conv1)
        bn1 = model.features[5](max0)
        tanh1 = model.features[6](bn1)
        
        conv2 = model.features[7](tanh1)
        bn2 = model.features[8](conv2)
        tanh2 = model.features[9](bn2)
        
        conv3 = model.features[10](tanh2)
        max1 = model.features[11](conv3)
        bn3 = model.features[12](max1)
        tanh3 = model.features[13](bn3)
        
        conv4 = model.features[14](tanh3)
        bn4 = model.features[15](conv4)
        tanh4 = model.features[16](bn4)
        
        conv5 = model.features[17](tanh4)
        max2 = model.features[18](conv5)
        bn5 = model.features[19](max2)
        tanh5 = model.features[20](bn5)

        tanh5 = tanh5.view(-1, 512*4*4)

        fc0 = model.classifier[0](tanh5)
        bn6 = model.classifier[1](fc0)
        tanh6 = model.classifier[2](bn6)

        fc1 = model.classifier[3](tanh6)
        bn7 = model.classifier[4](fc1)
        tanh7 = model.classifier[5](bn7)

        fc2 = model.classifier[6](tanh7)
        bn8 = model.classifier[7](fc2)
        soft = model.classifier[8](bn8)

#print(input_var[0][0])
#print('conv1', conv0[0][0][0])
#print('bn1', bn0[0][0][0])
#print('tanh1', tanh0[0][0][0])
#print('conv2', conv1[0][0][0])
#print('conv3', conv2[0][0][0])
#print('conv4', conv3[0][0][0])
#print('conv5', conv4[0][0][0])
#print('conv6', conv5[0][0][0])

print('tanh6')
for i in range(0, 16):
    print(tanh5[0][i])

print('fc1')
for i in range(0, 20):
    print(fc0[0][i])

print('fc2')
for i in range(0, 20):
    print(fc1[0][i])

print('fc3', fc2[0])
print('softmax', soft)

print('fc1 bias')
for i in range(0, 20):
    print(model.classifier[0].bias[i])
