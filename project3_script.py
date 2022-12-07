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
from scipy import signal

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
# Filter applied to each activity type
rest_data_filtered = p3m.filter_butter(rest_data_file)
relaxing_data_filtered = p3m.filter_butter(relaxing_data_file)
stress_rest_data_filtered = p3m.filter_butter(stress_rest_data_file)
physical_data_filtered = p3m.filter_butter(physical_data_file)

#Plot filter's impulse response and frequency response

fs = 500
dt = 1/fs
#create time array
t = np.arange(0, (len(rest_data_file))/fs , dt)
#create unit impulse
unit_impulse = signal.unit_impulse(len(t))
#get impulse response by sending unit impulse through filter
impulse_response = p3m.filter_butter(unit_impulse)

#plot impulse response
plt.figure(3, clear = True)
plt.subplot(1, 2, 1)
plt.plot(t, impulse_response)
plt.title('Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude?')

#plot frequency response
# use fft to put impulse response in frequency domain
frequency_butter_filter = fft.rfft(impulse_response)
# get frequency values on x axis
f = fft.rfftfreq(len(impulse_response), dt)

#plot frequency response
plt.subplot(1, 2, 2)
plt.plot(f, frequency_butter_filter)
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('X(f)')


# Plot data from rest activity before and after filter is applied

#plot before filter
plt.figure(4, clear = True)
plt.subplot(1, 2, 1)
plt.plot(rest_data_file)
plt.title("Rest Heart Rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")

#plot after filter
plt.subplot(1, 2, 2)
plt.plot(rest_data_filtered)
plt.title('Rest Heart Rate Filtered')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")


#%% Part 3: Detect Heartbeats
# Plot data for each activity's detected heartbeat times

# detect heart beats for rest activity
rest_heartbeat = p3m.detect_beats(rest_data_filtered, 1) #threshold = 1
# get times of heart beats
t_rest_heartbeat = np.arange(len(rest_data_filtered))
# plot rest data with heartbeat times
plt.figure('Heart Rate Data with Heartbeat Times', clear = True)
plt.subplot(2,2,1)
plt.plot(rest_data_filtered)
plt.scatter(t_rest_heartbeat[rest_heartbeat], rest_data_filtered[rest_heartbeat], c='green')
plt.title('Restful Activity Filtered\n w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)

#detect heart beats for relaxing activity
relaxing_heartbeat = p3m.detect_beats(relaxing_data_filtered, 1) #threshold = 1
#get times of heartbeats
t_relaxing_heartbeat = np.arange(len(relaxing_data_filtered))
#plot relaxing data with heartbeat times
plt.subplot(2,2,2)
plt.plot(relaxing_data_filtered)
plt.scatter(t_relaxing_heartbeat[relaxing_heartbeat], relaxing_data_filtered[relaxing_heartbeat], c='green')
plt.title('Relaxing Heart Rate\n Filtered w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)

#detect heart beats for stressful rest activity
stress_rest_heartbeat = p3m.detect_beats(stress_rest_data_filtered, 1) #threshold = 1
#get times of heartbeats
t_stress_rest_heartbeat = np.arange(len(stress_rest_data_filtered))
#plot relaxing data with heartbeat times
plt.subplot(2,2,3)
plt.plot(stress_rest_data_filtered)
plt.scatter(t_stress_rest_heartbeat[stress_rest_heartbeat], stress_rest_data_filtered[stress_rest_heartbeat], c='green')
plt.title('Mental Stress Heart Rate\n Filtered w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)

# detect heart beats for physical activity
physical_heartbeat = p3m.detect_beats(physical_data_filtered, 1) #threshold = 1
# get times of heart beats
t_physical_heartbeat = np.arange(len(physical_data_filtered))
# plot rest data with heartbeat times
plt.subplot(2,2,4)
plt.plot(physical_data_filtered)
plt.scatter(t_physical_heartbeat[physical_heartbeat], physical_data_filtered[physical_heartbeat], c='green')
plt.title('Physical Activity Heart Rate\n Filtered w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)


#%% Part 4: Calculate Heart Rate Variability
# Plot HRV measure for each activity in a bar graph (activity on x axis, HRV on y axis)

# Calculate an interpolated timecourse of IBI at regular intervals of dt=0.1 seconds

#%% Part 5: Get HRV Frequency Band Power
# Plot frequency domain magnitude in power

# Plot ratios of LF/HF in a bar graph