# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:44:31 2022

@author: haoyev5
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from d2l import torch as d2l
import torchvision
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import AverageMeter, natural_key, setup_default_logging, set_jit_legacy
from transformers import SwinForImageClassification
import ean
from transformers import AutoFeatureExtractor, VanForImageClassification
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import utils
import numpy as np


batch_size          = 32
num_classes         = 8
ddata_dir   = 'D:/data/FI/FI2'
train_dir, test_dir = 'train', 'valid'

num_epochs, lr, wd = 50, 5e-4, 5e-4
lr_period, lr_decay = 10, 0.1
#devices = d2l.try_all_gpus()
devices = [d2l.try_gpu(0)]

transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize((256,256)),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4,contrast=0.4,
                                       saturation=0.4),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# load data
train_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, train_dir), transform=transform_train)
if test_dir is not None:
    test_ds = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, test_dir), transform=transform_test)
else:
    test_ds = None
#test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, test_dir), flag=1)
train_iter = torch.utils.data.DataLoader(
    train_ds, batch_size, shuffle=True, drop_last=True)
print('train iter complete!')
if test_ds is not None:
    test_iter = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False, drop_last=True)
    print('test iter complete!')
else:
    test_iter = None
    print('No test iter! Go ahead---->')
   
############################################################################
loss = nn.CrossEntropyLoss(reduction="none")

net1 = ean.ean(pretrained=True)
net2 = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

class Get_Net(nn.Module):
    def __init__(self, net1, net2, num_classes=8, embedding=256):
        super().__init__()
        
        self.branch1 = net1
        self.branch2 = net2
        self.head = nn.Sequential(nn.Linear(2000, embedding),
                                  nn.GELU(),                    
                                  #nn.Dropout(.8),
                                  nn.Linear(embedding, num_classes))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x).logits
        y = torch.cat([x1,x2], 1)
        y = self.head(y)

        return y

x = torch.randn(2, 3, 224, 224)
net = Get_Net(net1, net2, num_classes=num_classes)
x = net(x)
print(x.shape)

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        for i, (features, labels) in enumerate(train_iter):
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
        time_s = "time %.2f sec" % (time.time() - start)
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            print(f'epoch{epoch+1}, train loss:{l / labels.shape[0]:.4f}, '
                  f'train acc:{acc / labels.shape[0] :.4f}, '
                  f'valid acc:{valid_acc:.4f}, {time_s}')
        else:
            print(f'epoch{epoch+1}, train loss:{l / labels.shape[0]:.4f}, '
                  f'train acc:{acc / labels.shape[0] :.4f},  {time_s}')
        scheduler.step()


train(net, train_iter, test_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

PATH='model.params'
torch.save(net.state_dict(), PATH)
############################################################################
# plot the confusion matrix
predict = []
labels = []
for data, label in test_iter:
    y_hat  = net(data.to(devices[0]))
    predict.extend(y_hat.argmax(axis=1).cpu().numpy())
    #label = label.asnumpy()
    labels.extend(label.cpu().numpy())

              
print("Number of test image:", len(labels))
#print('predict type:', type(predict))
#print('labels type', type(labels))

# 获取混淆矩阵
if num_classes == 2:
    classes = ['Negative', 'Positive']
elif num_classes == 6:
    classes = ['Anger', 'Disgust', 'Fear','Joy', 'Sadness', 'Surprise']
else:
    #classes = ['Amusement', 'Awe', 'Contentment', 'Excitement', 
    #           'Anger', 'Disgust', 'Fear', 'Sadness']
    classes = ['Amusement', 'Contentment', 'Awe', 'Excitement', 
               'Fear', 'Sadness', 'Disgust', 'Anger']
print(classes)
#cm = confusion_matrix(labels, predict)
cm = utils.confusion_matrix(predict, labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 进行归一化处理，得到概率值，对角线元素即为召回率
utils.plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', classes, title='confusion matrix')
utils.plot_Matrix(cm_normalized, classes)






