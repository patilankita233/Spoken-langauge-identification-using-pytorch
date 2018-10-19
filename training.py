#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:57:08 2018

@author: krishna
"""
import torch
import numpy as np
from utils.compute_filterbank import compute_filterbank_energy
import random
from utils.create_minibatch import create_batches_rnd
import torch.nn as nn
from models.ConvNet import ConvNet

random.seed(2018)


training_files =  [line.rstrip('\n') for line in open('data_utils/train_list.txt')]
testing_files = [line.rstrip('\n') for line in open('data_utils/test_list.txt')]
N_train_files = len(training_files)
N_test_files = len(testing_files)
#batch_size=4
#batch_list = training_files[:batch_size]
#input_batch,labels = create_batch(batch_list,global_length)

### Hyper parameters
batch_size=32
learning_rate = 0.01
num_classes=6
N_epochs = 30

#######
fs = 48000
window_len = 3*fs
input_shape=[40,297]
###### Model configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
seed=1234

#####
### Shuffling the list to randomize the data
random.shuffle(training_files)
######
N_batches = 10

for epoch in range(N_epochs):
    loss_sum=0
    err_sum=0
    acc_sum=0
    for i in range(N_batches):
        input_data,labels = create_batches_rnd(batch_size,training_files,N_train_files,window_len,0.2)
        inputs = torch.reshape(input_data,(batch_size,1,input_shape[0],input_shape[1]))
        inputs = inputs.to(device)
        labels = labels.to(device) 
        output = model(inputs)
        loss = criterion(output,labels.long())
        prediction=torch.max(output,dim=1)[1]
        err = torch.mean((prediction!=labels.long()).float())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.item())
        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
        acc_sum  = acc_sum+torch.mean((prediction==labels.long()).float())
    loss_tot=loss_sum/N_batches
    err_tot=err_sum/N_batches
    tot_acc = acc_sum/N_batches
    print('Total training loss----->'+str(loss_tot))
    print('Total error loss----->'+str(err_tot))
    print('Total training accuracy----->'+str(tot_acc))