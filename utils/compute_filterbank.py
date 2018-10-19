#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:47:44 2018

@author: krishna
"""

'''
Returns filterbank matrix for a given audio file
'''


import numpy as np
import speechpy
from speechpy.feature import lmfe

def compute_filterbank_energy(signal,fs,frame_length=0.025,frame_stride=0.01):
    #fs,audio_data = wav.read(wav_path)
    filterbank_energy = lmfe(signal,fs,frame_length,frame_stride,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    #filterbank_energy_inv = np.transpose(filterbank_energy)
    normalized =  speechpy.feature.processing.cmvn(filterbank_energy, variance_normalization=True)
    
    return np.transpose(normalized)

