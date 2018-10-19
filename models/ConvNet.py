#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:57:08 2018

@author: krishna
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,30,kernel_size=(5,1),stride=(1,1),padding=(2,1)),
                nn.BatchNorm2d(30),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)))
        
        self.layer2 = nn.Sequential(nn.Conv2d(30,60,kernel_size=(5,1),stride=(1,1),padding=(2,1)),
                nn.BatchNorm2d(60),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)))
        
        self.layer3 = nn.Sequential(nn.Conv2d(60,80,kernel_size=(5,1),stride=(1,1),padding=(2,1)),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)))
        
        
        self.layer4 = nn.Sequential(nn.Conv2d(80,100,kernel_size=(5,1),stride=(1,1),padding=(2,1)),
                nn.BatchNorm2d(100),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2)))
        
        self.layer5 = nn.Sequential(nn.Conv2d(100,40,kernel_size=(5,1),stride=(1,1),padding=(2,1)),
                nn.BatchNorm2d(40),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,1)))
        
        
        self.fc1 = nn.Linear(40*40*21, 5000)
        self.fc2 = nn.Linear(5000,1000)
        self.fc3 = nn.Linear(1000, num_classes)




    def forward(self,x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        out = layer5_out.reshape(layer5_out.size(0),-1)
        fc1_out = F.dropout(F.relu(self.fc1(out)),p=0.3)
        fc2_out = F.dropout(F.relu(self.fc2(fc1_out)),p=0.3)
        fc3_out = F.relu(self.fc3(fc2_out))
        return fc3_out
    
    
