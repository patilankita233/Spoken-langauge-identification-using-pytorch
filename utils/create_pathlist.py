#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:55:54 2018

@author: krishna
"""
'''
This scripts creates 2 files in data_utils directory which contains the paths of the wavefiles
'''



import os
import glob




#data_dir = sys.argv[1]
data_dir = '/Users/krishna/Krishna/Langauge_ID_CNN/'
if not os.path.exists('data_utils/'):
    os.makedirs('data_utils/')


training_folder = data_dir+'Processed_Data/train/'
testing_folder = data_dir+'Processed_Data/test/'

#### Create list files for training folder
fid = open(os.getcwd()+'/data_utils/train_list.txt','w')


all_langs = glob.glob(training_folder+'/*/')
for lang_path in all_langs:
    all_files = glob.glob(lang_path+'/*.wav')
    for file_path in all_files:
        fid.write(file_path+'\n')
fid.close()

##### Create list files for testing
fid = open(os.getcwd()+'/data_utils/test_list.txt','w')

all_langs = glob.glob(testing_folder+'/*/')
for lang_path in all_langs:
    all_files = glob.glob(lang_path+'/*.wav')
    for file_path in all_files:
        fid.write(file_path+'\n')
fid.close()

###############################
## Creating the label files
### 





if os.path.exists(os.getcwd()+'/data_utils/train_list.txt'):
    print('List files are created')
else:
    print('List files are not created...check the code')
    
    




