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
def detect_beats(signal, threshold, fs):
    potential_beat = np.where(signal >= threshold)[0] 
    #Get indicies of every first value above the threshold
    beat_locations = potential_beat[np.insert(np.diff(potential_beat) > 1, 0, True)] #Insert a 0 at the start because frist value in potential beat is a beat
    
    time_general = np.arange(len(signal))/fs
    beat_time = time_general[beat_locations]
    
    plt.plot(time_general, signal)
    plt.scatter(beat_time, signal[beat_locations], c='green')
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.tight_layout(pad = 3)
    
    
    return beat_locations, beat_time


#%% Part 4: Calculate Heart Rate Variability
# Create function to calculate the inter-beat intervals from detected heartbeats
def calculate_ibis(beat_locations):
    dt = .01
    fs = 10
    interpolated_time = np.arange(0, (beat_locations.max()), dt)
    ibi_values = np.diff(beat_locations, axis = 0)
    beat_times = beat_locations[0:-1] /fs
    interpolated_ibi = np.interp(interpolated_time, beat_times, ibi_values)
    hrv = np.std(interpolated_ibi)
    
    return interpolated_ibi, hrv
    
    
    #or index in beat_time:
        #ibi_values are the differences between the times of heartbeats
        #ibi_values = np.zeros(len(beat_time) - 1)
        #ibi_values = np.diff(beat_time, axis=0)
        #ibi_times = np.zeros(len(signal_heartbeat) -1)
        #ibi_times = signal_heartbeat[ibi_values]
        
    #return ibi_values
# Create function to calculate one HRV measure for each activity
#def calculate_hrv(ibi_values):
    #hrv = np.std(ibi_values)
    
    #return(hrv)

#interpolate IBI at dt=0.1
#def interpolate_data(beat_time, ibi_values, dt=0.1):
    #beat_time = beat_time[0:-1]
    #interpolated_time = np.arange(0,np.max(beat_time),dt)
    #interpolated_ibi = np.interp(interpolated_time, beat_time, ibi_values)
    
    #return interpolated_ibi


#%% Part 5: Get HRV Frequency Band Power
# Create function to calculate the frequency domain magnitude of each activity’s IBI timecourse signal
def frequency_filter(ibi_values, dt = 0.1):
    
    frequency_fft = fft.rfft(ibi_values - np.mean(ibi_values))
    power = np.square(np.abs(frequency_fft))
    frequency = fft.rfftfreq(len(ibi_values), dt)
    
    low_freq_index = np.zeros(len(frequency))
    low_freq_index[(frequency >= 0.04) & (frequency <= 0.15)] = 1
    low_freq_index = low_freq_index.astype(int)
    low_freq = frequency[(frequency >= 0.04) & (frequency <= 0.15)]
    low_power_index = power * low_freq_index
    low_power = low_power_index[low_power_index >0]
   
    high_freq_index = np.zeros(len(frequency))
    high_freq_index[(frequency >= 0.15) & (frequency <= 0.4)] = 1
    high_freq_index = high_freq_index.astype(int)
    high_freq = frequency[(frequency >= 0.15) & (frequency <= 0.4)]
    high_power_index = power * high_freq_index
    high_power = high_power_index[high_power_index >0]
    
    
    return frequency, power, low_freq, low_power, high_freq, high_power
    
    
    
    
    #return frequency, power, low_freq, low_power, high_freq, high_power
#Create function to extract the mean power in the LF and HF frequency bands & calculate the LF/HF ratio for each

