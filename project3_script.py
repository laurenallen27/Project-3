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
duration = 300 #seconds data counted for
fs = 500
dt = 1/fs

rest_data_file = p3m.load_data('rest_data (1).txt', duration, fs)
relaxing_data_file = p3m.load_data('on_phone_data (1).txt', duration, fs)
stress_rest_data_file = p3m.load_data('stressful_rest (1).txt', duration, fs)
physical_data_file = p3m.load_data('wallsit_data (1).txt', duration, fs)

#get time arrays for plots
t = np.arange(0, (len(rest_data_file))/fs , dt)

# Plot of 5 minutes sitting at rest
plt.figure(1, clear = True)
plt.subplot(2,2,1)
plt.plot(t, rest_data_file)
plt.title("Rest heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.tight_layout(pad = 3)

# Plot of 5 minutes of relaxing activity
plt.subplot(2,2,2)
plt.plot(t,relaxing_data_file)
plt.title("Relaxing heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(9,14)
plt.tight_layout(pad = 3)

# Plot of 5 minutes of mentally stressful activity
plt.subplot(2,2,3)
plt.plot(t,stress_rest_data_file)
plt.title("Mental srtess heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.tight_layout(pad = 3)

# Plot of 5 minutes of physically stressful activity
plt.subplot(2,2,4)
plt.plot(t,physical_data_file)
plt.title("Physical stress heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(47,52)
plt.tight_layout(pad = 3)

# Plot of 4 activity recordings concatenated together
# concatenate 4 signals
concatenated_data = np.concatenate((rest_data_file, relaxing_data_file, stress_rest_data_file, physical_data_file))
plt.figure(2, clear = True)  
plt.plot(concatenated_data)  
plt.title("Concatenated signal") 
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")       
plt.tight_layout(pad = 3)               

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
plt.plot(impulse_response)
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
plt.plot(t, rest_data_file)
plt.title("Rest Heart Rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)

#plot after filter
plt.subplot(1, 2, 2)
plt.plot(t,rest_data_filtered)
plt.title('Rest Heart Rate Filtered')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)


#%% Part 3: Detect Heartbeats
# Plot data for each activity's detected heartbeat times

# detect heart beats for rest activity
rest_heartbeat = p3m.detect_beats(rest_data_filtered, 40) #threshold = 40
# get times of heart beats
t_rest_heartbeat = np.arange(len(rest_data_filtered))
# plot rest data with heartbeat times
plt.figure('Heart Rate Data with Heartbeat Times', clear = True)
plt.subplot(2,2,1)
plt.plot(t,rest_data_filtered)
plt.scatter(t_rest_heartbeat[rest_heartbeat], rest_data_filtered[rest_heartbeat], c='green')
plt.title('Restful Activity Filtered\n w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)
plt.xlim(0,5)

#detect heart beats for relaxing activity
relaxing_heartbeat = p3m.detect_beats(relaxing_data_filtered, 1) #threshold = 1
#get times of heartbeats
t_relaxing_heartbeat = np.arange(len(relaxing_data_filtered))
#plot relaxing data with heartbeat times
plt.subplot(2,2,2)
plt.plot(t,relaxing_data_filtered)
plt.scatter(t_relaxing_heartbeat[relaxing_heartbeat], relaxing_data_filtered[relaxing_heartbeat], c='green')
plt.title('Relaxing Heart Rate\n Filtered w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)
plt.xlim(9,14)

#detect heart beats for stressful rest activity
stress_rest_heartbeat = p3m.detect_beats(stress_rest_data_filtered, 1) #threshold = 1
#get times of heartbeats
t_stress_rest_heartbeat = np.arange(len(stress_rest_data_filtered))
#plot relaxing data with heartbeat times
plt.subplot(2,2,3)
plt.plot(t,stress_rest_data_filtered)
plt.scatter(t_stress_rest_heartbeat[stress_rest_heartbeat], stress_rest_data_filtered[stress_rest_heartbeat], c='green')
plt.title('Mental Stress Heart Rate\n Filtered w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)
plt.xlim(0,5)

# detect heart beats for physical activity
physical_heartbeat = p3m.detect_beats(physical_data_filtered, 1) #threshold = 1
# get times of heart beats
t_physical_heartbeat = np.arange(len(physical_data_filtered))
# plot rest data with heartbeat times
plt.subplot(2,2,4)
plt.plot(t,physical_data_filtered)
plt.scatter(t_physical_heartbeat[physical_heartbeat], physical_data_filtered[physical_heartbeat], c='green')
plt.title('Physical Activity Heart Rate\n Filtered w/ Heartbeat Times')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.tight_layout(pad = 3)
plt.xlim(47,52)


#%% Part 4: Calculate Heart Rate Variability
# Plot HRV measure for each activity in a bar graph (activity on x axis, HRV on y axis)
# call IBI values for all data
rest_ibi = p3m.calculate_ibi(rest_heartbeat)
relaxing_ibi = p3m.calculate_ibi(relaxing_heartbeat)
stress_rest_ibi = p3m.calculate_ibi(stress_rest_heartbeat)
physical_ibi = p3m.calculate_ibi(physical_heartbeat)


# call hrv value for all data
rest_hrv = p3m.calculate_hrv(rest_ibi)
relaxing_hrv = p3m.calculate_hrv(relaxing_ibi)
stress_rest_hrv = p3m.calculate_hrv(stress_rest_ibi)
physical_hrv = p3m.calculate_hrv(physical_ibi)

#plot hrv on bar graph
#x = [('Rest', 'Relaxing', 'Mental Stress', 'Physical Activity')]
#y = [(rest_hrv, relaxing_hrv, stress_rest_hrv, physical_hrv)]
#plt.figure(6, clear = True)
#plt.bar(x,y)
n = 4
r = np.arange(4)
width = 1
plt.figure(6, clear = True)
plt.bar(1, rest_hrv, color = 'b', width = width, label = 'Rest')
plt.bar(2 , relaxing_hrv, color = 'g', width = width, label = 'Relaxing')
plt.bar(3 , stress_rest_hrv, color = 'r', width = width, label = 'Stressful Rest')
plt.bar(4 , physical_hrv, color = 'y', width = width, label = 'Physical Activity')
plt.legend(loc = 'upper left')

# Calculate an interpolated timecourse of IBI at regular intervals of dt=0.1 seconds
#fs is 1/dt = 10hz, whent aking FT of interpokated signal, hughe # at 0 and small right after
#comes from high frequency sin wave fluctuations centered around dc offset bc mean of sigmal is not zero
#dc offset is 0hz sin wave, way bigger than the rest so it trumps everything else
#can subtract the mean of the signal before taking the transform, or can take plot and zoom in on y axis to see fluctuations
#freq response has y range determined by fs, nyquist frequency is 1/2 of sampling frequency, determined by interpolated signal freq


#%% Part 5: Get HRV Frequency Band Power
# Plot frequency domain magnitude in power

# Plot ratios of LF/HF in a bar graph