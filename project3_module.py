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

#%% Part 1: Collect and Load Data
# Create function to load data for 4 different activity categories
def load_data(input_file):
    data_file = np.loadtxt(input_file, dtype = float)
    
    return data_file
    

#%% Part 2: Filter Your Data
# Create a function to apply to filter to each dataset

#%% Part 3: Detect Heartbeats
# Create function to detect heartbeats in each dataset

#%% Part 4: Calculate Heart Rate Variability
# Create function to calculate the inter-beat intervals from detected heartbeats

# Create function to calculate one HRV measure for each activity

#%% Part 5: Get HRV Frequency Band Power
# Create function to calculate the frequency domain magnitude of each activity’s IBI timecourse signal

#Create function to extract the mean power in the LF and HF frequency bands & calculate the LF/HF ratio for each

