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
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(rest_data_file)

# Plot of 5 minutes of relaxing activity
plt.subplot(2,2,2)
plt.plot(relaxing_data_file)

# Plot of 5 minutes of mentally stressful activity
plt.subplot(2,2,3)
plt.plot(stress_rest_data_file)

# Plot of 5 minutes of physically stressful activity
plt.subplot(2,2,4)
plt.plot(physical_data_file)

# Plot of 4 activity recordings concatenated together
# concatenate 4 signals
concatenated_data = np.concatenate((rest_data_file, relaxing_data_file, stress_rest_data_file, physical_data_file))
plt.figure(2)  
plt.plot(concatenated_data)                         

#%% Part 2: Filter Your Data
# Create filter to get rid of low frequency drift and high frequency fuzz

#Plot filter's impulse response and frequency response

# Plot data from one activity before and after filter is applied

#%% Part 3: Detect Heartbeats
# Plot data for each activity's detected heartbeat times

#%% Part 4: Calculate Heart Rate Variability
# Plot HRV measure for each activity in a bar graph (activity on x axis, HRV on y axis)

# Calculate an interpolated timecourse of IBIat regular intervalsof dt=0.1 seconds

#%% Part 5: Get HRV Frequency Band Power
# Plot frequency domain magnitude in power

# Plot ratios of LF/HF in a bar graph