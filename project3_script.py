#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:52:31 2022

Project 3: Script
This script calls the functions from the module and produces arrays and plots which
contain values specific to the ECG voltage data collected during four different
types of activities: rest, relaxing activity, mentally stressful activity, and a 
physically stressful activity all over the duration of 5 minutes. The script produces 
plots of the unfiltered ECG data for each activity, the concatenated data, the impulse and 
frequency responses of our bandpass filter, and filtered ECG data for one activity.
Additionally, plots of the ECG data with heartbeat times, an HRV bar graph, plots of
the frequency domain magnitude of IBIs with LF and HF bands, as well as an HRV ratio
bar graph are produced. These graphs allow us to examine data and see how each activity
affects the ANS. 

@authors: laurenallen, altagodfrey
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import project3_module as p3m
from scipy import signal

#%% Part 1: Collect and Load Data
# define values for duration, fs, and dt so that files can be clipped to 5 minutes
duration = 300 #seconds data counted for (5minutes*60seconds)
fs = 500 #sampling frequency for arduino
dt = 1/fs

#Call the load data function for each activity
rest_data_file = p3m.load_data('rest_data (1).txt', duration, fs)
relaxing_data_file = p3m.load_data('on_phone_data (1).txt', duration, fs)
stress_rest_data_file = p3m.load_data('stressful_rest (1).txt', duration, fs)
physical_data_file = p3m.load_data('wallsit_data (1).txt', duration, fs)

#get time arrays for x-axis of plots
#since all data files are the same length, this time variable can be used for all data.
t = np.arange(0, (len(rest_data_file))/fs , dt)

# Plot of ECG activity for 5 minutes sitting at rest
plt.figure('ECG Data Unfiltered', clear = True)
plt.subplot(4,1,1) #create 4 rows of subplots
plt.plot(t, rest_data_file)
# annotate plot
plt.title("Rest ECG Data")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
# 5 seconds of data
plt.xlim(0,5)
plt.grid()
plt.tight_layout()

# Plot of ECG activity for 5 minutes of relaxing activity
plt.subplot(4,1,2)
plt.plot(t,relaxing_data_file)
# annotate plot
plt.title("Relaxing ECG Data")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
# 5 seconds of data
plt.xlim(9,14) #chose this range by dragging plot to see when waves were best seen
plt.grid()
plt.tight_layout()

# Plot of ECG activity fot 5 minutes of mentally stressful activity
plt.subplot(4,1,3)
plt.plot(t,stress_rest_data_file)
# annotate plot
plt.title("Mental Stress ECG Data")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5)
plt.grid()
plt.tight_layout()

# Plot of ECG activity for 5 minutes of physically stressful activity
plt.subplot(4,1,4)
plt.plot(t,physical_data_file)
# annotate plot
plt.title("Physical Stress ECG Data")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(47,52) #chose this range by dragging plot to see when waves were best seen
plt.grid()
plt.tight_layout()
plt.savefig('Unfiltered ECG Data')

# Plot of 4 activity recordings concatenated together
# concatenate 4 signals
concatenated_data = np.concatenate((rest_data_file, relaxing_data_file, stress_rest_data_file, physical_data_file))
plt.figure('Concatenated ECG Data', clear = True)  
plt.plot(concatenated_data)  
plt.title("Concatenated signal") 
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")      
plt.grid()
plt.tight_layout(pad = 3) #add more space between plots         
#save figure
plt.savefig('Concatenated Signal')    

#%% Part 2: Filter Your Data
# Call function to apply bandpass filter to each activity data file
rest_data_filtered = p3m.filter_butter(rest_data_file)
relaxing_data_filtered = p3m.filter_butter(relaxing_data_file)
stress_rest_data_filtered = p3m.filter_butter(stress_rest_data_file)
physical_data_filtered = p3m.filter_butter(physical_data_file)

#Plot filter's impulse response and frequency response
#create unit impulse using scipy function
#input is the # of samples we want in the output, which is the length of time array
unit_impulse = signal.unit_impulse(len(t)) 
#get impulse response by sending unit impulse through filter
impulse_response = p3m.filter_butter(unit_impulse)

#plot impulse response
plt.figure('Impulse & Frequency Response', clear = True)
plt.subplot(1, 2, 1) #create subplot with 2 columns
plt.plot(impulse_response)
# annotate plot
plt.title('Impulse Response')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (A.U.)')
plt.xscale('log') #log scale for x axis
plt.grid()
plt.tight_layout()


#plot frequency response
# use fourier transform to put impulse response in frequency domain
frequency_butter_filter = fft.rfft(impulse_response)
# get corresponding frequency values for x axis
f = fft.rfftfreq(len(impulse_response), dt)

#plot frequency response
plt.subplot(1, 2, 2)
plt.plot(f, frequency_butter_filter)
#annotate plot
plt.title('Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (A.U)')
plt.xscale('log')
plt.grid()
plt.tight_layout()
# save figure
plt.savefig('Impulse & Frequency Response')


# Plot data from rest activity before and after filter is applied
# plot before filter
plt.figure('Rest ECG Data Before & After Filter', clear = True)
plt.subplot(1, 2, 1)
plt.plot(t, rest_data_file)
# annotate plot
plt.title("Rest ECG Data w/o Filter")
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5) #zoom in on 5 seconds
plt.grid()

#plot after filter
plt.subplot(1, 2, 2)
plt.plot(t,rest_data_filtered)
#annotate plot
plt.title('Rest ECG Data Filtered')
plt.ylabel("Voltage (mV)")
plt.xlabel("Time (s)")
plt.xlim(0,5) #zoom in on 5 seconds
plt.grid()
plt.tight_layout()
#save figure
plt.savefig('Rest ECG Data Filtered')


#%% Part 3: Detect Heartbeats
# Plot data for each activity's detected heartbeat times
# plot rest data with heartbeat times
plt.figure('ECG Data with Heartbeat Times', clear = True)
plt.subplot(4,1,1) #create subplot with 4 rows
# call function to get rest heartbeat locations and heartbeat times, then plot them
rest_heartbeat, rest_heartbeat_time = p3m.detect_beats(rest_data_filtered, 40, fs) #threshold = 40
plt.title('Restful Activity Filtered\n w/ Heartbeat Times')
plt.xlim(0,5) #zoom in on 5 seconds
plt.grid()

#detect heart beats for relaxing activity
plt.subplot(4,1,2)
# call function to get relaxing heartbeat locations and heartbeat times, then plot them
relaxing_heartbeat, relaxing_heartbeat_time = p3m.detect_beats(relaxing_data_filtered, 40, fs) #threshold = 40
plt.title('Relaxing Activity Filtered\n w/ Heartbeat Times')
plt.xlim(9,14) #zoom in on 5 seconds
plt.grid()

#detect heart beats for stressful rest activity
plt.subplot(4,1,3)
# call function to get mental stress heartbeat locations and heartbeat times, then plot them
stress_rest_heartbeat, stress_rest_heartbeat_time = p3m.detect_beats(stress_rest_data_filtered,40, fs) #threshold = 40
plt.title('Mentally Stressful Activity Filtered\n w/ Heartbeat Times')
plt.xlim(0,5) #zoom in on 5 seconds
plt.grid()

# detect heart beats for physical activity
plt.subplot(4,1,4)
# call function to get physical stres heartbeat locations and heartbeat times, then plot them
physical_heartbeat, physical_heartbeat_time = p3m.detect_beats(physical_data_filtered, 40, fs) #threshold = 40
plt.title('Physical Activity Filtered\n w/ Heartbeat Times')
plt.xlim(47,52) #zoom in on 5 seconds
plt.grid()
#save figure
plt.savefig('Filtered ECG Data with Heartbeats')


#%% Part 4: Calculate Heart Rate Variability
# call HRV values and interpolated ibi timecourses for all data (will only use HRV in this part)
interpolated_rest, rest_hrv = p3m.calculate_ibis(rest_heartbeat, rest_heartbeat_time)
interpolated_relaxing, relaxing_hrv = p3m.calculate_ibis(relaxing_heartbeat, relaxing_heartbeat_time)
interpolated_stress_rest, stress_rest_hrv = p3m.calculate_ibis(stress_rest_heartbeat, stress_rest_heartbeat_time)
interpolated_physical, physical_hrv = p3m.calculate_ibis(physical_heartbeat, physical_heartbeat_time)

# Plot HRV measure for each activity in a bar graph (activity on x axis, HRV on y axis)
x = np.array(['Rest', 'Relaxing', 'Mental Stress', 'Physical Activity'])
y = np.array([rest_hrv, relaxing_hrv, stress_rest_hrv, physical_hrv])
plt.figure('HRV Bar Graph', clear = True)
plt.bar(x,y)
plt.ylabel('HRV')
plt.title('Heart Rate Variability for Various Activities')
#save figure
plt.savefig('HRV Bar Graph')



#%% Part 5: Get HRV Frequency Band Power
# Plot frequency domain magnitude in units of power (PSD)
plt.figure('FFT Spectrum', clear = True)
# plot rest data
plt.subplot(4,1,1) #create subplot with 4 rows
#call function to plot frequency domain magnitude w/ LF & HF bands for rest data
rest_frequency, rest_power, rest_low_frequency, rest_low_power, rest_high_frequency, rest_high_power = p3m.frequency_filter(interpolated_rest, dt)
plt.ylim(0,0.25*(10**9))
plt.title('FFT Spectrum - Rest')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()

#plot relaxing data
plt.subplot(4,1,2)
#call function to plot frequency domain magnitude w/ LF & HF bands for relaxing data
relaxing_frequency, relaxing_power, relaxing_low_frequency, relaxing_low_power, relaxing_high_frequency, relaxing_high_power = p3m.frequency_filter(interpolated_relaxing, dt)
plt.title('FFT Spectrum - Relax')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()

#plot mental stress data
plt.subplot(4,1,3)
#call function to plot frequency domain magnitude w/ LF & HF bands for mental stress data
stress_rest_frequency, stress_rest_power, stress_rest_low_frequency, stress_rest_low_power, stress_rest_high_frequency, stress_rest_high_power = p3m.frequency_filter(interpolated_stress_rest, dt)
plt.ylim(0,0.25*(10**8))
plt.title('FFT Spectrum - Stress Rest')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()

#plot physical stress data
plt.subplot(4,1,4)
#call function to plot frequency domain magnitude w/ LF & HF bands for physical stress data
physical_frequency, physical_power, physical_low_frequency, physical_low_power, physical_high_frequency, physical_high_power = p3m.frequency_filter(interpolated_physical, dt)
plt.ylim(0,1*(10**9))
plt.title('FFT Spectrum - Physical Stress')
plt.ylabel('Power (s^2/Hz)')
plt.xlabel('Frequency(Hz)')
plt.tight_layout()
#save figure
plt.savefig('FFT Spectrum')

# Plot ratios of LF/HF in a bar graph
# call function to retrieve ratios of mean LF/HF for each activity
rest_ratio = p3m.extract_mean_power(rest_low_power, rest_high_power)
relaxing_ratio = p3m.extract_mean_power(relaxing_low_power, relaxing_high_power)
stress_rest_ratio = p3m.extract_mean_power(stress_rest_low_power, stress_rest_high_power)
physical_ratio = p3m.extract_mean_power(physical_low_power, physical_high_power)

#plot hrv ratios on bar graph
x = np.array(['Rest', 'Relaxing', 'Mental Stress', 'Physical Activity'])
y = np.array([rest_ratio, relaxing_ratio, stress_rest_ratio, physical_ratio])
plt.figure('HRV Ratio Bar Graph', clear = True)
plt.bar(x,y)
plt.ylabel('HRV Ratio')
plt.title('LF/HF Ratio for various activities')
#Save figure
plt.savefig('HRV Ratio Bar Graph')



