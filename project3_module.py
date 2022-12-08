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
def load_data(input_file, duration, fs):
    data_file = np.loadtxt(input_file, dtype = float)
    #data_file = data_file * 5/1024
    data_file = data_file[0:duration*fs]
    
    return data_file
    
#%% Part 2: Filter Your Data
# Create a function to apply bandpass butterworth filter to each dataset
def filter_butter(signal):
    # define lowcut freq, highcut freq, and sampling freq
    lowcut = 0.5 #30bpm in bps
    highcut = 2.5 #150bpm in bps
    fs = 500
    
    #get values for frequency band
    nyq = fs * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    
    #set order
    order = 2
    
    #get coefficients
    b, a = scipy.signal.butter(order, (low,high), 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)
    # return filtered signal
    return y
    
    
#%% Part 3: Detect Heartbeats
# Create function to detect heartbeats in each dataset
def detect_beats(signal, threshold):
    potential_beat = np.where(signal >= threshold)[0]
    #Get indicies of every first value above the threshold
    beat_locations = potential_beat[np.insert(np.diff(potential_beat) > 1, 0, True)] #Insert a 0 at the start because frist value in potential beat is a beat
    
    return beat_locations

#%% Part 4: Calculate Heart Rate Variability
# Create function to calculate the inter-beat intervals from detected heartbeats
def calculate_ibi(signal_heartbeat):
    for index in signal_heartbeat:
        ibi_values = np.zeros(len(signal_heartbeat) - 1)
        ibi_values = np.diff(signal_heartbeat, axis=0)
        #ibi_times = np.zeros(len(signal_heartbeat) -1)
        #ibi_times = signal_heartbeat[ibi_values]
        
    return ibi_values
# Create function to calculate one HRV measure for each activity
def calculate_hrv(ibi_values):
    hrv = np.std(ibi_values)
    
    return(hrv)

#interpolate IBI at dt=0.1
def interpolate_data(beat_time, ibi_values, dt=0.1):
    interpolated_time = np.arange(0,np.max(beat_time),dt)
    interpolated_data = np.interp(interpolated_time, beat_time, ibi_values)
    
    return interpolated_time, interpolated_data


#%% Part 5: Get HRV Frequency Band Power
# Create function to calculate the frequency domain magnitude of each activity’s IBI timecourse signal

#Create function to extract the mean power in the LF and HF frequency bands & calculate the LF/HF ratio for each

