#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:51:14 2022

Project 3: Module 

@author: laurenallen, altagodfrey
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import scipy 
from scipy.signal import filtfilt

#%% Part 1: Collect and Load Data
# Create function to load data for 4 different activity categories
def load_data(input_file):
    data_file = np.loadtxt(input_file, dtype = float)
    
    return data_file
    

#%% Part 2: Filter Your Data
# Create a function to apply to filter to each dataset

#fs=0.8
#def filter_butter(signal, passband, stopband, fs=fs):
    
    #order, wn = scipy.signal.buttord(passband, stopband, 4, 40)
    #sos = scipy.signal.butter(5, fs, btype = 'bandpass')
    #butter_filter = sos.filtfilt
    
    #return butter_filter


def filter_butter(signal):
    low = 1
    high = 1.7
    fs = 500
    
    nyq = fs * 2
    low = low / nyq
    high = high / nyq
    
    order = 2
    
    b, a = scipy.signal.butter(order, (low,high), 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)
    
    return y
#%% Part 3: Detect Heartbeats
# Create function to detect heartbeats in each dataset

#%% Part 4: Calculate Heart Rate Variability
# Create function to calculate the inter-beat intervals from detected heartbeats

# Create function to calculate one HRV measure for each activity

#%% Part 5: Get HRV Frequency Band Power
# Create function to calculate the frequency domain magnitude of each activity’s IBI timecourse signal

#Create function to extract the mean power in the LF and HF frequency bands & calculate the LF/HF ratio for each

