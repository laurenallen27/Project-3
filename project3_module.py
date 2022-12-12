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
    '''
    A function to load the data file and clip it so that for a given duration &
   sampling frequency, each data file contains the same number of samples. 

    Parameters
    ----------
    input_file : string
        Name of the txt file to be loaded
    duration : integer
        The time in seconds that the data should be collected for
    fs : integer
        The sampling frequency in Hz or 1/s

    Returns
    -------
    data_file : array of size (x,) where x is the number of samples
        1D array containing the ecg voltage data at a given sampling frequency

    '''
    data_file = np.loadtxt(input_file, dtype = float)
    #data_file = data_file * 5/1024
    data_file = data_file[0:duration*fs]
    
    return data_file
    
#%% Part 2: Filter Your Data
# Create a function to apply bandpass butterworth filter to each dataset
def filter_butter(signal):
    '''
    A function to create a bandpass filter which removes noise and artifacts from
    a given signal. 

    Parameters
    ----------
    signal : array of size (x,) where x is the number of samples
        1D array containing the ecg voltage data at a given sampling frequency

    Returns
    -------
    filtered_signal : array of size (x,) where x is the number of samples
        1D array containing the filtered ECG data 

    '''
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
    filtered_signal = scipy.signal.filtfilt(b, a, signal, axis=0)
    # return filtered signal
    return filtered_signal
    
    
#%% Part 3: Detect Heartbeats
# Create function to detect heartbeats in each dataset
def detect_beats(signal, threshold, fs):
    '''
    A function to detect when beats occur in a signal by determining if the
    signal value passes a certain threshold. 

    Parameters
    ----------
    signal : Array of integers.
        An array .
    threshold : Integer.
        Specified value to identify the QRS wave complex.
    fs : integer
        The sampling frequency in Hz or 1/s

    Returns
    -------
    beat_locations : Array of Integers
        Representing the values of the signal that exceed the specified signal, 
        marking the location of the QRS wave in the sample signal .
    beat_time : Array of Integers
        Representing the times at which the beat exceeded the specified threshold.

    '''
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
def calculate_ibis(beat_locations, beat_time):
    '''
    

    Parameters
    ----------
    beat_locations : Array of Integers
        Representing the values of the signal that exceed the specified signal, 
        marking the location of the QRS wave in the sample signal .
    beat_time : Array of Integers
        Representing the times at which the beat exceeded the specified threshold.

    Returns
    -------
    interpolated_ibi : Array of Floats
        A collection of the interpolated inter beat intervals for the identified beats.
    hrv : Array of Floats
        A value representing the standard deviation of the inter beat interval
        of the specified signal.

    '''
    dt = 0.1
    fs = 10
    interpolated_time = np.arange(0, beat_locations.max(), dt)
    ibi_values = np.diff(beat_time, axis = 0)
    beat_times = beat_locations[1:] / fs
    interpolated_ibi = np.interp(interpolated_time, beat_times, ibi_values)
    hrv = np.std(interpolated_ibi)
    
    return interpolated_ibi, hrv
    

#%% Part 5: Get HRV Frequency Band Power
# Create function to calculate the frequency domain magnitude of each activity’s IBI timecourse signal
def frequency_filter(ibi_values, dt = 0.1):
    '''
    

    Parameters
    ----------
    ibi_values : Array of 
        DESCRIPTION.
    dt : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    frequency : TYPE
        DESCRIPTION.
    power : TYPE
        DESCRIPTION.
    low_freq : TYPE
        DESCRIPTION.
    low_power : TYPE
        DESCRIPTION.
    high_freq : TYPE
        DESCRIPTION.
    high_power : TYPE
        DESCRIPTION.

    '''
    
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
    
    plt.plot(frequency, power)
    plt.plot(low_freq, low_power, alpha = 0.7, color='y')
    plt.fill_between(low_freq, low_power, alpha = 0.5, color='y')
    plt.plot(high_freq, high_power, alpha = 0.7, color='g')
    plt.fill_between(high_freq, high_power, alpha = 0.5, color='g')
    plt.xlim(0.03, 0.4)
    
    
    return frequency, power, low_freq, low_power, high_freq, high_power
    
#Create function to extract the mean power in the LF and HF frequency bands & calculate the LF/HF ratio for each
def extract_mean_power(low_power, high_power):
    '''
    

    Parameters
    ----------
    low_power : TYPE
        DESCRIPTION.
    high_power : TYPE
        DESCRIPTION.

    Returns
    -------
    ratio : TYPE
        DESCRIPTION.

    '''
    mean_lf = np.mean(low_power)
    mean_hf = np.mean(high_power)
    
    ratio = mean_lf / mean_hf
    
    return ratio
    


