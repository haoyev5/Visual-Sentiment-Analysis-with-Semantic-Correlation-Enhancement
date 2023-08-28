# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 10:46:15 2022

@author: haoyev5
"""
import collections
import math
import os
import shutil
import pandas as pd
from mxnet import gluon, init, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

"""
将数据集的图像移动到按类别明明的文件夹下：
test
   |--1
   |--2
   |--3
   ···
   ···
   ···
"""


data_dir = 'E:/FI'


def read_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()
    tokens = [l.rstrip().split('/') for l in lines]
    return dict(((name, label) for label, name in tokens))

labels = read_labels(os.path.join(data_dir, 'TestImages.txt'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))


def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
    

def reorg_img(data_dir, labels):
    """"读取label中的文件，并移动到指定位置"""
    n = len(labels)
    names = labels.keys()
    names = list(names)
    for name in names:
        label = labels[name]
        fname = os.path.join(data_dir, 'Images', name)
        copyfile(fname, os.path.join(data_dir, 'valid', label))
    
    return n

def reorg_data(data_dir):
    labels = read_labels(os.path.join(data_dir, 'TestImages.txt'))
    reorg_img(data_dir, labels)

reorg_data(data_dir)

        
    
