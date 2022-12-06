#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:52:31 2022

Project 3: Script

@author: laurenallen, altagodfrey
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import project3_module as p3m

#%% Part 1: Collect and Load Data
# Call the load data function for each activity
rest_data_file = p3m.load_data('rest_data (1).txt')
relaxing_data_file = p3m.load_data('on_phone_data (1).txt')
stress_rest_data_file = p3m.load_data('stressful_rest (1).txt')
physical_data_file = p3m.load_data('wallsit_data (1).txt')


# Plot of 5 minutes sitting at rest
plt.figure(1, clear = True)
plt.subplot(2,2,1)
plt.plot(rest_data_file)
plt.title("Rest heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")

# Plot of 5 minutes of relaxing activity
plt.subplot(2,2,2)
plt.plot(relaxing_data_file)
plt.title("Relaxing heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")

# Plot of 5 minutes of mentally stressful activity
plt.subplot(2,2,3)
plt.plot(stress_rest_data_file)
plt.title("Mental srtess heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")

# Plot of 5 minutes of physically stressful activity
plt.subplot(2,2,4)
plt.plot(physical_data_file)
plt.title("Physical stress heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")

# Plot of 4 activity recordings concatenated together
# concatenate 4 signals
concatenated_data = np.concatenate((rest_data_file, relaxing_data_file, stress_rest_data_file, physical_data_file))
plt.figure(2, clear = True)  
plt.plot(concatenated_data)  
plt.title("Concatenated signal") 
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")                      

#%% Part 2: Filter Your Data
# Create filter to get rid of low frequency drift and high frequency fuzz

#Plot filter's impulse response and frequency response

#plot impulse response

fs = 500
t = np.arange(0, len(rest_data_file), 1/fs)
unit_step = np.zeros(len(t))
unit_step[t>=0]=1

impulse_response = p3m.filter_butter(unit_step)

plt.figure(3, clear = True)
plt.subplot(1, 2, 1)
plt.plot(impulse_response)

#plot frequency response

frequency_butter_filter = fft.rfft(impulse_response)

plt.subplot(1, 2, 2)
plt.plot(frequency_butter_filter)


# Plot data from one activity before and after filter is applied

#plot before filter
plt.figure(4, clear = True)
plt.subplot(1, 2, 1)
plt.plot(rest_data_file)
plt.title("Rest heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Frequency")

#plot after filter

rest_data_filter =  p3m.filter_butter(rest_data_file)

plt.subplot(1, 2, 2)
plt.plot(rest_data_filter)



#%% Part 3: Detect Heartbeats
# Plot data for each activity's detected heartbeat times

#%% Part 4: Calculate Heart Rate Variability
# Plot HRV measure for each activity in a bar graph (activity on x axis, HRV on y axis)

# Calculate an interpolated timecourse of IBIat regular intervalsof dt=0.1 seconds

#%% Part 5: Get HRV Frequency Band Power
# Plot frequency domain magnitude in power

# Plot ratios of LF/HF in a bar graph