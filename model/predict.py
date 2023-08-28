# -*- coding: utf-8 -*-
"""
Created on Thu May 19 07:59:17 2022

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
import van
from transformers import AutoFeatureExtractor, VanForImageClassification
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import utils
import numpy as np
from sklearn.metrics import confusion_matrix


batch_size          = 32
num_classes         = 6
#data_dir   = 'D:/data/pcnn/agg5'  
data_dir   = 'D:/data/emotion6'   #artphoto2 emotion6
#data_dir   = 'D:/data/FI/FI'
train_dir, test_dir = 'train', 'valid'


PATH='model4emotion6.params'
#devices = d2l.try_all_gpus()
devices = [d2l.try_gpu(0)]



transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((448,448)),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# load test data

test_ds = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, test_dir), transform=transform_test)

test_iter = torch.utils.data.DataLoader(
    test_ds, batch_size, shuffle=False, drop_last=True)
print('test iter complete!')

   
############################################################################
loss = nn.CrossEntropyLoss(reduction="none")

net1 = van.van_base(pretrained=False)
net2 = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

class Get_Net(nn.Module):
    def __init__(self, net1, net2, num_classes=1000, embedding=256):
        super().__init__()
        
        self.branch1 = net1
        self.branch2 = net2
        self.head = nn.Sequential(nn.Linear(2000, embedding),
                                  nn.ReLU(),
                                  #nn.Dropout(.8),
                                  nn.Linear(embedding, num_classes))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x).logits
        y = torch.cat([x1,x2], 1)
        y = self.head(y)

        return y

net = Get_Net(net1, net2, num_classes=num_classes)
net.load_state_dict(torch.load(PATH, map_location='cpu'))
net= net.to(devices[0])

'''
x = torch.randn(2, 3, 224, 224)
x = net(x)
print(x.shape)
'''

############################################################################
# plot the confusion matrix
print("Start the forecasting process")
predict = []
labels = []
for data, label in test_iter:
    y_hat  = net(data.to(devices[0]))
    predict.extend(y_hat.argmax(axis=1).cpu().numpy())
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
cm = confusion_matrix(predict, labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 进行归一化处理，得到概率值，对角线元素即为召回率
utils.plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', classes, title='confusion matrix')
utils.plot_Matrix(cm_normalized, classes)
