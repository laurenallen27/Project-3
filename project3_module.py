#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:51:14 2022

Project 3: Module 
This module contains functions which manipulate ECG data recorded using Arduino and
wearable ECG sensors. The data is loaded into an array of a specified length, and filtered
using a bandpass butterworth filter. There are then functions which detect when heartbeats
occur, calculate a set of IBIs at evenly spaced intervals, and then calculate heart rate
variability. The interpolated IBIs are then tranformed into the frequency domain and plotted
in units of power with specified high and low frequency bands. Finally, the mean power
in these bands is extracted and used to calculate the LF/HF ratio, which will ultimately be 
used to estimate ANS activity.

Sources used:https://www.youtube.com/watch?v=juYqcck_GfU
Used video as guidance on how to construct the bandpass filter in part 2.

@authors: laurenallen, altagodfrey
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
    data_file : array of floats size (x,) where x is the number of samples
        1D array containing the ecg voltage data at a given sampling frequency

    '''
    # load the .txt file into an array
    data_file = np.loadtxt(input_file, dtype = float)
    # clip data file so that it is a constant length
    data_file = data_file[0:duration*fs]
    #return the trimmed array of data
    return data_file
    
#%% Part 2: Filter Your Data
# Create a function to apply bandpass butterworth filter to each dataset
def filter_butter(signal):
    '''
    A function to create a bandpass filter which removes noise and artifacts from
    a given signal. 

    Parameters
    ----------
    signal : array of floats size (x,) where x is the number of samples
        1D array containing the ecg voltage data at a given sampling frequency

    Returns
    -------
    filtered_signal : array of floats size (x,) where x is the number of samples
        1D array containing the filtered ECG data 

    '''
    # define lowcut freq, highcut freq, and sampling freq
    lowcut = 0.5 #30bpm in bps, very low range of a heartrate
    highcut = 2.5 #150bpm in bps, high range of a heartrate
    fs = 500 #arduino sampling frequency
    
    #get values for frequency band cutoff values
    nyq = fs * 0.5 #nyquist frequncy is half of the sampling frequency
    low = lowcut / nyq #low cutoff frequency
    high = highcut / nyq #high cutoff frequency
    
    #set order
    order = 2 #order or 2 used in many examples online
    
    #get coefficients to input into butterworth bandpass filter
    b, a = scipy.signal.butter(order, (low,high), 'bandpass', analog=False)
    filtered_signal = scipy.signal.filtfilt(b, a, signal, axis=0)
    # return filtered signal with less noise and artifacts
    return filtered_signal
    
#%% Part 3: Detect Heartbeats
# Create function to detect heartbeats in each dataset
def detect_beats(signal, threshold, fs):
    '''
    A function to detect when beats occur in a signal by determining if the
    sample voltage passes a certain threshold. 

    Parameters
    ----------
    signal : array of floats size (x,) where x is the number of samples
        1D array containing the ecg voltage data at a given sampling frequency
    threshold : Integer.
        Specified value to identify the QRS wave complex.
    fs : integer
        The sampling frequency in Hz or 1/s

    Returns
    -------
    beat_locations : Array of integers size (x,) where x is the # of beats detected
        Contains the samples where the voltage exceeds the threshold, 
        marking the location of the QRS waves in the signal.
    beat_time : Array of floats size (x,) where x is the # of beats detected
        Representing the times at which the beat exceeded the specified threshold.

    '''
    # get the samples where the voltage value is greater than or equal to a threshold value
    potential_beat_index = np.where(signal >= threshold)[0] 
    #Get indicies of every first value above the threshold, insert a 0 at the start because first value in potential beat is a beat
    beat_locations = potential_beat_index[np.insert(np.diff(potential_beat_index) > 1, 0, True)] 
    
    #create a time array the same length as the signal voltage data
    time_general = np.arange(len(signal))/fs
    # calculate the times at each location where a beat occurs
    beat_time = time_general[beat_locations]
    
    #plot the signal in time domain with the beat locations specified as dots
    plt.plot(time_general, signal)
    plt.scatter(beat_time, signal[beat_locations], c='green')
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.tight_layout(pad = 3)
    
    #return array containing the samples (voltages) and times at which a beat occurs 
    return beat_locations, beat_time


#%% Part 4: Calculate Heart Rate Variability
# Create function to calculate the inter-beat intervals from detected heartbeats, interpolate the IBIs, and calculate HRV
def calculate_ibis(beat_locations, beat_time):
    '''
    A function to calculate the inter-beat intervals from detected heartbeats, then 
    interpolate the IBIs at times of known heartbeats to estimate the IBIs at unknown,
    evenly spaced times, and finally calculate the HRV or the difference between the 
    interpolated IBI timecourse values. 

    Parameters
    ----------
    beat_locations : Array of integers size (x,) where x is the # of beats detected
        Contains the samples where the voltage exceeds the threshold, 
        marking the location of the QRS waves in the signal.
    beat_time : Array of floats size (x,) where x is the # of beats detected
        Representing the times at which the beat exceeded the specified threshold.

    Returns
    -------
    interpolated_ibi : Array of Floats size (x,) where x is the # of interpolated IBI values 
        A collection of the interpolated inter-beat-intervals for the identified beats.
    hrv : Float
        A value representing the standard deviation of the inter beat interval
        of the specified signal

    '''
    dt = 0.1 #interpolated time array should be evenly spaced at intervals of .1 second
    fs = 10 #fs = 1/dt
    # create time array to interpolate IBIs
    interpolated_time = np.arange(0, beat_locations.max(), dt)
    # calculate inter-beat-intervals from detected heartbeats
    ibi_values = np.diff(beat_time, axis = 0) #difference between the times of each beat
    #beat times corresponding to the detected beats at a sampling frequency of 10Hz
    beat_times = beat_locations[1:] / fs 
    # interpolation using IBIs at known (heartbeat) times to estimate the IBIs at unknown (regularly spaced) times
    interpolated_ibi = np.interp(interpolated_time, beat_times, ibi_values)
    
    # calculate heart rate variability, the difference in times between beats
    hrv = np.std(interpolated_ibi) #standard deviation of interpolated interbeat intervals
    
    #return the interpolated IBI timecourse and HRV value for signal
    return interpolated_ibi, hrv
    

#%% Part 5: Get HRV Frequency Band Power
# Create function to calculate the frequency domain magnitude of each activity’s IBI timecourse signal
def frequency_filter(ibi_values, dt = 0.1):
    '''
    A function to calculate the magnitude of an IBI timecourse (units of power) in the 
    frequency domain using a fast fourier transform, then filter the data to obtain
    high and low frequency bands. 
    
    Parameters
    ----------
    ibi_values : Array of floats size (x,) where x is the # of IBIs after interpolation
        The time between each beat at an evenly spaced interval of dt = .1
    dt : float, optional
        The amount of time between each IBI value. The default is 0.1.

    Returns
    -------
    frequency : Array of floats size (x,) where x is the # of samples obtained after 
    the transform
        Array containing the frequency values corresponding to the magnitude values of 
        the IBI timecourse
    power : Array of floats size (x,) where x is the # of samples obtained after 
    the transform
        Array containing the magnitude of IBI timecourse signal in the frequency
        domain in units of power
    low_freq : Array of floats size (x,) where x is # of freq values within LF band
        Frequency values within the range of the low frequency band (0.04-.15Hz)
    low_power : Array of floats size (x,) where x is # of freq values within LF band
        Corresponding power values within the range of the low frequency band (0.04 - .15Hz)
    high_freq : Array of floats size (x,) where x is # of freq values within HF band
        Frequency values within the range of the high frequency band (.15 - .4Hz)
    high_power : Array of floats size (x,) where x is # of freq values within HF band
        Corresponding power values within the range of the high frequency band (.15-.4Hz)

    '''
    # calculate IBI timecourse in frequency domain
    #subtract mean of the signal to account for high freq sin wave fluctuations around dc offset
    frequency_fft = fft.rfft(ibi_values - np.mean(ibi_values)) 
    # convert to units of power, square the absolute value of the frequency
    power = np.square(np.abs(frequency_fft))
    # calculate corresponding frequency values 
    frequency = fft.rfftfreq(len(ibi_values), dt)
    
    #initialize array for low frequency band 
    low_freq_index = np.zeros(len(frequency))
    # set array values where frequency is within .04-.15 Hz equal to one, used for bool indexing
    low_freq_index[(frequency >= 0.04) & (frequency <= 0.15)] = 1
    low_freq_index = low_freq_index.astype(int) #index values must be integers
    # get low frequency band frequency values
    low_freq = frequency[(frequency >= 0.04) & (frequency <= 0.15)]
    # keep power values that fall within the low freq range
    low_power_index = power * low_freq_index
    #get low frequency band power values
    low_power = low_power_index[low_power_index >0]
   
    #initialize array for low frequency band 
    high_freq_index = np.zeros(len(frequency))
    # set array values where frequency is within .15-.4 Hz equal to one
    high_freq_index[(frequency >= 0.15) & (frequency <= 0.4)] = 1
    high_freq_index = high_freq_index.astype(int)
    # get high frequency band frequency values
    high_freq = frequency[(frequency >= 0.15) & (frequency <= 0.4)]
    # keep power values that fall within the high freq range
    high_power_index = power * high_freq_index
    #get high frequency band power values
    high_power = high_power_index[high_power_index >0]
    
    #plot frequency domain magnitude in units of power for given IBI timecourse
    plt.plot(frequency, power)
    #plot low frequency band
    plt.plot(low_freq, low_power, alpha = 0.7, color='y')
    plt.fill_between(low_freq, low_power, alpha = 0.5, color='y') #shade below line
    # plot high frequency band
    plt.plot(high_freq, high_power, alpha = 0.7, color='g')
    plt.fill_between(high_freq, high_power, alpha = 0.5, color='g') #shade below line
    #zoom in on frequency bands
    plt.xlim(0,0.4)
    #return arrays containing frequency, power, as well as freq and power data for LF and Hf bands
    return frequency, power, low_freq, low_power, high_freq, high_power
    
#Create function to extract the mean power in the LF and HF frequency bands & calculate the LF/HF ratio for each
def extract_mean_power(low_power, high_power):
    '''
    Function to extract the mean power of the high and low frequency bands, then 
    take the ratio of mean LF power to mean HF power.

    Parameters
    ----------
    low_power : Array of floats size (x,) where x is # of freq values within LF band
        Power values within the range of the low frequency band (0.04 - .15Hz)
    high_power : Array of floats size (x,) where x is # of freq values within HF band
        Power values within the range of the high frequency band (.15-.4Hz)

    Returns
    -------
    ratio : float
        The ratio of the mean LF band power to the mean HF band power, AKA the HRV ratio.

    '''
    #extract mean power in LF band
    mean_lf = np.mean(low_power)
    #extract mean power in HF band
    mean_hf = np.mean(high_power)
    
    #calculare ratio of mean LF power to mean HF power (HRV ratio)
    ratio = mean_lf / mean_hf
    
    #return HRV ratio
    return ratio
    


