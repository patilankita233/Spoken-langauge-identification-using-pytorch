#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:48:35 2018

@author: krishna
"""
'''
Gnerates minibatchs of specific minibatch of filterbank data

'''


import numpy as np
import torch
from torch.autograd import Variable
import soundfile as sf
from utils.compute_filterbank import compute_filterbank_energy



lang_dict = {'German':0,'Italian':1,'English':2,'Portuguese':3,'French':4,'Spanish':5}




def get_labels(audio_filepath):
    label  = lang_dict[audio_filepath.split('/')[-2]]
    return label




def create_batches_rnd(batch_size,wav_lst,N_snt,wlen,fact_amp=0.2):
    
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    #sig_batch=np.zeros([batch_size,wlen])
    #lab_batch=np.zeros(batch_size)
  
    feat_matrix=[]
    label_matrix=[]
    
    snt_id_arr=np.random.randint(N_snt, size=batch_size)
    
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

    for i in range(batch_size):
         
         #filterbank_out = compute_filterbank_energy(wav_lst[snt_id_arr[i]])
        
         signal,fs = sf.read(wav_lst[snt_id_arr[i]])
         
         snt_len=signal.shape[0]
         if wlen>=snt_len:
             snt_beg=0
         else: 
            snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
         snt_end=snt_beg+wlen
         if snt_end>=snt_len:
             signal = np.append(signal,np.zeros((1,snt_end-snt_len))[0])
        
         cut_signal = signal[snt_beg:snt_end]*rand_amp_arr[i]
         if len(cut_signal)<wlen:
             signal = np.append(signal,np.zeros((1,wlen-len(cut_signal)))[0])
             
         filterbank_out = compute_filterbank_energy(cut_signal,fs)
         label_out = get_labels(wav_lst[snt_id_arr[i]])
         feat_matrix.append(filterbank_out)
         label_matrix.append(label_out)
         #plt.imshow(filterbank_out,cmap='jet')
         
         #sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
         #lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]


    #print(np.asarray(feat_matrix).shape)
    #print(np.asarray(label_matrix).shape)
    
    if torch.cuda.is_available():
        inp=Variable(torch.from_numpy(np.asarray(feat_matrix)).float().cuda().contiguous())
        lab=Variable(torch.from_numpy(np.asarray(label_matrix)).float().cuda().contiguous())
    else:
        inp=Variable(torch.from_numpy(np.asarray(feat_matrix)).float())
        lab=Variable(torch.from_numpy(np.asarray(label_matrix)).float())
    
    return inp,lab




    #inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    #lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
  
    #return inp,lab 



### Dummy
'''
batch_size=10
wav_lst = [line.rstrip('\n') for line in open('data_utils/train_list.txt','r')]
N_snt = len(wav_lst)
wlen = 3*fs
'''

#batch_size=20
#wav_lst = training_files
#N_snt=N_train_files,
#wlen = window_len
