import torch
import models
import numpy as np
from torch import nn
from torchsummary import summary as summary

model = models.vgg_cifar10_binary
model_config = {'input_size': None, 'dataset': 'cifar10'}
model = model(**model_config)
print("created model with configuration: %s"%(model_config))
model = model.to(torch.device('cuda'))

checkpoint = torch.load("./results/vgg_cifar10_binary/checkpoint.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
print("loaded checkpoint (epoch %s)"%(checkpoint['epoch']))

model.train(False)
summary(model, (3, 32, 32))

model = model.to(torch.device('cpu'))
for name, param in model.named_parameters():
    print(name, param.size())
    param = param.detach().numpy().reshape(param.size())
    fp = open('./weights/'+name+'.txt', 'w')
    if(name=='features.0.weight' or name=='features.3.weight' or
       name=='features.7.weight' or name=='features.10.weight' or
       name=='features.14.weight' or name=='features.17.weight'):
        for i in range(0, len(param)):                      # N (전체 개수)
            for j in range(0, len(param[0])):               # C (채널 개수)
                for k in range(0, len(param[0][0])):        # H (height)
                    for l in range(0, len(param[0][0][0])): # W (width)
                        if(param[i][j][k][l]==1.0):
                            fp.write(str(1)+'\n')
                        elif(param[i][j][k][l]==-1.0):
                            fp.write(str(-1)+'\n')
                        else:
                            fp.write(str('%.12f'%(param[i][j][k][l]))+'\n')
    elif(name=='classifier.0.weight' or name=='classifier.3.weight' or name=='classifier.6.weight'):
        for i in range(0, len(param)):
            for j in range(0, len(param[0])):
                if(param[i][j]==1.0):
                    fp.write(str(1)+'\n')
                elif(param[i][j]==-1.0):
                    fp.write(str(-1)+'\n')
                else:
                    fp.write(str('%.12f'%(param[i][j]))+'\n')
    else:
        for i in range(0, len(param)):
            if(param[i]==1.0):
                fp.write(str(1)+'\n')
            elif(param[i]==-1.0):
                fp.write(str(-1)+'\n')
            else:
                fp.write(str('%.12f'%(param[i]))+'\n')
    fp.close()

fp1 = open('./weights/features.1.running_mean.txt', 'w')
fp2 = open('./weights/features.1.running_var.txt', 'w')
param1 = model.features[1].running_mean
param2 = model.features[1].running_var
print('features.1.running_mean', param1.size())
print('features.1.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/features.5.running_mean.txt', 'w')
fp2 = open('./weights/features.5.running_var.txt', 'w')
param1 = model.features[5].running_mean
param2 = model.features[5].running_var
print('features.5.running_mean', param1.size())
print('features.5.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/features.8.running_mean.txt', 'w')
fp2 = open('./weights/features.8.running_var.txt', 'w')
param1 = model.features[8].running_mean
param2 = model.features[8].running_var
print('features.8.running_mean', param1.size())
print('features.8.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/features.12.running_mean.txt', 'w')
fp2 = open('./weights/features.12.running_var.txt', 'w')
param1 = model.features[12].running_mean
param2 = model.features[12].running_var
print('features.12.running_mean', param1.size())
print('features.12.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/features.15.running_mean.txt', 'w')
fp2 = open('./weights/features.15.running_var.txt', 'w')
param1 = model.features[15].running_mean
param2 = model.features[15].running_var
print('features.15.running_mean', param1.size())
print('features.15.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/features.19.running_mean.txt', 'w')
fp2 = open('./weights/features.19.running_var.txt', 'w')
param1 = model.features[19].running_mean
param2 = model.features[19].running_var
print('features.19.running_mean', param1.size())
print('features.19.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/classifier.1.running_mean.txt', 'w')
fp2 = open('./weights/classifier.1.running_var.txt', 'w')
param1 = model.classifier[1].running_mean
param2 = model.classifier[1].running_var
print('classifier.1.running_mean', param1.size())
print('classifier.1.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/classifier.4.running_mean.txt', 'w')
fp2 = open('./weights/classifier.4.running_var.txt', 'w')
param1 = model.classifier[4].running_mean
param2 = model.classifier[4].running_var
print('classifier.4.running_mean', param1.size())
print('classifier.4.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()

fp1 = open('./weights/classifier.7.running_mean.txt', 'w')
fp2 = open('./weights/classifier.7.running_var.txt', 'w')
param1 = model.classifier[7].running_mean
param2 = model.classifier[7].running_var
print('classifier.7.running_mean', param1.size())
print('classifier.7.running_var', param2.size())
param1 = param1.detach().numpy().reshape(param1.size())
param2 = param2.detach().numpy().reshape(param2.size())
for i in range(0, len(param1)):
    fp1.write(str('%.12f'%(param1[i]))+'\n')
    fp2.write(str('%.12f'%(param2[i]))+'\n')
fp1.close()
fp2.close()
