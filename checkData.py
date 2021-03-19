from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from utils.dataset import MovingMNISTDataset
from configs.config_3x3_16_3x3_32_3x3_64 import config

import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
import torch


train_dataset = MovingMNISTDataset(config, split='train')

#print(type(train_dataset[1]))

#print(len(train_dataset[0][0][0]))



#print(train_dataset[500][1][9])

#print(train_dataset[500][1][7].shape)
#print(train_dataset[0][1].shape)
#print(train_dataset[0][0].shape)

this_label = torch.squeeze(train_dataset[500][1][9]).numpy()
#print(np.max(this_label))
#print(np.min(this_label))

#cv2.imwrite("label.png", this_label*255)

#print(train_dataset[0][1][1][0][0])
#print(len(train_dataset[0][1][1][0][0]))


print(len(train_dataset))
print(type(train_dataset[0][0]))
print(train_dataset[1][0].shape)
print(train_dataset[1][1].shape)

print(train_dataset[0][1])
print(train_dataset[0][0][1].shape)