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

# #normalize data
# rest_normal = fft.rfft(rest_data_file - np.mean(rest_data_file))

#get time arrays for plots\
#since all data files are the same length, this time variable can be used for all data.
t = np.arange(0, (len(rest_data_file))/fs , dt)

# Plot of 5 minutes sitting at rest
plt.figure(1, clear = True)
plt.subplot(4,1,1)
plt.plot(t, rest_data_file)
plt.title("Rest heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.grid()
plt.tight_layout()

# Plot of 5 minutes of relaxing activity
plt.subplot(4,1,2)
plt.plot(t,relaxing_data_file)
plt.title("Relaxing heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(9,14)
plt.grid()
plt.tight_layout()

# Plot of 5 minutes of mentally stressful activity
plt.subplot(4,1,3)
plt.plot(t,stress_rest_data_file)
plt.title("Mental srtess heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.grid()
plt.tight_layout()

# Plot of 5 minutes of physically stressful activity
plt.subplot(4,1,4)
plt.plot(t,physical_data_file)
plt.title("Physical stress heart rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(47,52)
plt.grid()
plt.tight_layout()
plt.savefig('Unfiltered heart beat')

# Plot of 4 activity recordings concatenated together
# concatenate 4 signals
concatenated_data = np.concatenate((rest_data_file, relaxing_data_file, stress_rest_data_file, physical_data_file))
plt.figure(2, clear = True)  
plt.plot(concatenated_data)  
plt.title("Concatenated signal") 
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")      
plt.grid()
plt.tight_layout(pad = 3)           
plt.savefig('Concatenated signal')    

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
plt.ylabel('H(t)')
plt.xscale('log')
plt.grid()
plt.tight_layout()


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
plt.ylabel('Magnitude (A.U)')
plt.xscale('log')
plt.grid()
plt.tight_layout()
plt.savefig('Filter')


# Plot data from rest activity before and after filter is applied

#plot before filter
plt.figure(4, clear = True)
plt.subplot(1, 2, 1)
plt.plot(t, rest_data_file)
plt.title("Rest Heart Rate")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.grid()

#plot after filter
plt.subplot(1, 2, 2)
plt.plot(t,rest_data_filtered)
plt.title('Rest Heart Rate Filtered')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.grid()
plt.tight_layout()
plt.savefig('Rest Heart Rate Filtered')


#%% Part 3: Detect Heartbeats
# Plot data for each activity's detected heartbeat times

# detect heart beats for rest activity

# plot rest data with heartbeat times
plt.figure('Heart Rate Data with Heartbeat Times', clear = True)
plt.subplot(4,1,1)
rest_heartbeat, rest_heartbeat_time = p3m.detect_beats(rest_data_filtered, 40, fs) #threshold = 40
plt.title('Restful Activity Filtered\n w/ Heartbeat Times')
plt.xlim(0,5)
plt.grid()

#detect heart beats for relaxing activity
plt.subplot(4,1,2)
relaxing_heartbeat, relaxing_heartbeat_time = p3m.detect_beats(relaxing_data_filtered, 40, fs) #threshold = 40
plt.title('Relaxing Activity Filtered\n w/ Heartbeat Times')
plt.xlim(9,14)
plt.grid()

#detect heart beats for stressful rest activity
plt.subplot(4,1,3)
stress_rest_heartbeat, stress_rest_heartbeat_time = p3m.detect_beats(stress_rest_data_filtered,40, fs) #threshold = 40
plt.title('Mentally Stressful Activity Filtered\n w/ Heartbeat Times')
plt.xlim(0,5)
plt.grid()

# detect heart beats for physical activity
plt.subplot(4,1,4)
physical_heartbeat, physical_heartbeat_time = p3m.detect_beats(physical_data_filtered, 40, fs) #threshold = 40
plt.title('Physical Activity Filtered\n w/ Heartbeat Times')
plt.xlim(47,52)
plt.grid()
plt.savefig('Filtered Heart Beats')


#%% Part 4: Calculate Heart Rate Variability
# Plot HRV measure for each activity in a bar graph (activity on x axis, HRV on y axis)
# call HRV values for all data
interpolated_rest, rest_hrv = p3m.calculate_ibis(rest_heartbeat, rest_heartbeat_time)
interpolated_relaxing, relaxing_hrv = p3m.calculate_ibis(relaxing_heartbeat, relaxing_heartbeat_time)
interpolated_stress_rest, stress_rest_hrv = p3m.calculate_ibis(stress_rest_heartbeat, stress_rest_heartbeat_time)
interpolated_physical, physical_hrv = p3m.calculate_ibis(physical_heartbeat, physical_heartbeat_time)

#plot hrv on bar graph
x = np.array(['Rest', 'Relaxing', 'Mental Stress', 'Physical Activity'])
y = np.array([rest_hrv, relaxing_hrv, stress_rest_hrv, physical_hrv])
plt.figure(6, clear = True)
plt.bar(x,y)
plt.ylabel('HRV')
plt.title('Heart Rate Variability for Various Activities')
plt.savefig('HRV Bar Graph')


# Calculate an interpolated timecourse of IBI at regular intervals of dt=0.1 seconds
#fs is 1/dt = 10hz, whent aking FT of interpokated signal, hughe # at 0 and small right after
#comes from high frequency sin wave fluctuations centered around dc offset bc mean of sigmal is not zero
#dc offset is 0hz sin wave, way bigger than the rest so it trumps everything else
#can subtract the mean of the signal before taking the transform, or can take plot and zoom in on y axis to see fluctuations
#freq response has y range determined by fs, nyquist frequency is 1/2 of sampling frequency, determined by interpolated signal freq


#%% Part 5: Get HRV Frequency Band Power
# Plot frequency domain magnitude in power

plt.figure(8, clear = True)

plt.subplot(4,1,1)
rest_frequency, rest_power, rest_low_frequency, rest_low_power, rest_high_frequency, rest_high_power = p3m.frequency_filter(interpolated_rest, dt)
plt.ylim(0,0.25*(10**9))
plt.title('FFT Spectrum - Rest')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()

plt.subplot(4,1,2)
relaxing_frequency, relaxing_power, relaxing_low_frequency, relaxing_low_power, relaxing_high_frequency, relaxing_high_power = p3m.frequency_filter(interpolated_relaxing, dt)
plt.title('FFT Spectrum - Relax')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()

plt.subplot(4,1,3)
stress_rest_frequency, stress_rest_power, stress_rest_low_frequency, stress_rest_low_power, stress_rest_high_frequency, stress_rest_high_power = p3m.frequency_filter(interpolated_stress_rest, dt)
plt.ylim(0,0.25*(10**8))
plt.title('FFT Spectrum - Stress Rest')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()

plt.subplot(4,1,4)
physical_frequency, physical_power, physical_low_frequency, physical_low_power, physical_high_frequency, physical_high_power = p3m.frequency_filter(interpolated_physical, dt)
plt.ylim(0,1*(10**9))
plt.title('FFT Spectrum - Physical Stress')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()
plt.savefig('FFT Spectrum')

# Plot ratios of LF/HF in a bar graph
rest_ratio = p3m.extract_mean_power(rest_low_power, rest_high_power)
relaxing_ratio = p3m.extract_mean_power(relaxing_low_power, relaxing_high_power)
stress_rest_ratio = p3m.extract_mean_power(stress_rest_low_power, stress_rest_high_power)
physical_ratio = p3m.extract_mean_power(physical_low_power, physical_high_power)

#plot hrv on bar graph
x = np.array(['Rest', 'Relaxing', 'Mental Stress', 'Physical Activity'])
y = np.array([rest_ratio, relaxing_ratio, stress_rest_ratio, physical_ratio])
plt.figure(9, clear = True)
plt.bar(x,y)
plt.ylabel('HRV Ratio')
plt.title('LF/HF Ratio for various activities')
plt.savefig('HRV Ratio Bar Graph')


