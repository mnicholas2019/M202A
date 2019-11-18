# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:22:33 2019

@author: Aidan
"""

import csv
#import warnings
import os
import audio_metadata
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.stats import binned_statistic
from Training import Activity

ACCEL_FILE_NAME = "ACCELEROMETER--org.md2k.motionsense--MOTION_SENSE_HRV--RIGHT_WRIST.csv"
GYRO_FILE_NAME = "GYROSCOPE--org.md2k.motionsense--MOTION_SENSE_HRV--RIGHT_WRIST.csv"
MARKER_FILE_NAME = "MARKER--org.md2k.mcerebrum--PHONE.csv"
ESENSE_FILE_NAME = "esense_data.txt"
AUDIO_FILE_NAME = "audio.wav"

WINDOW_LENGTH = 1000 # How long each training activity is
STRIDE_LENGTH = 100 # How often to create a new training activity
ESENSE_SAMPLE_RATE = 20
WRIST_SAMPLE_RATE = 2000

AUDIO_SAMPLE_RATE = 8000 
AUDIO_MAX_FREQ = 2000 # Maximum audio frequency to analyze
AUDIO_BINS = 100 # Number of Frequency bins for fft audio

DOWNSAMPLE_ESENSE = 1000 / ESENSE_SAMPLE_RATE
DOWNSAMPLE_WRIST = 1000 / WRIST_SAMPLE_RATE

plt.style.use('seaborn-whitegrid')

ACTIVITIES = {
              "Other": 0,
              "Eating chips": 1,
              "Drinking": 2,
              "Smoking": 3,
              "Head scratch": 4
              }

def merge_sensor_data_stride(esense_data, wrist_acc_data, wrist_gryo_data, audio_data, activity_data):
    training_activities = []    
    num_activities = 0
    for activity in activity_data:
        label = int(activity[0])
        start = activity[1]
        end = activity[2]
        
        num_sub_activities = int((end - start - WINDOW_LENGTH) / STRIDE_LENGTH)
        num_activities += num_sub_activities
        
        for i in range(num_sub_activities):
            end = start + WINDOW_LENGTH
            
            esense_trimmed = esense_data[np.logical_and(esense_data[:, 0] >= start,  esense_data[:, 0] <= end)]
            wrist_acc_trimmed = wrist_acc_data[np.logical_and(wrist_acc_data[:, 0] >= start,  wrist_acc_data[:, 0] <= end)]
            wrist_gyro_trimmed = wrist_gryo_data[np.logical_and(wrist_gryo_data[:, 0] >= start,  wrist_gryo_data[:, 0] <= end)]
            audio_trimmed = audio_data[np.logical_and(audio_data[:, 0] >= start,  audio_data[:, 0] <= end)]
            
            audio_fft = binned_audio_fft(audio_trimmed[:, 1], AUDIO_SAMPLE_RATE, AUDIO_MAX_FREQ, AUDIO_BINS)
        
            training_activities.append(Activity(label, 
                                                esense_trimmed,  
                                                wrist_acc_trimmed,
                                                wrist_gyro_trimmed,
                                                audio_fft[1]
                                                ))
            start += STRIDE_LENGTH
    print(num_activities)
    return training_activities

def merge_sensor_data(esense_data, wrist_acc_data, wrist_gryo_data, audio_data, activity_data):
    training_activities = []    
    for activity in activity_data:
        label = int(activity[0])
        start = activity[1]
        end = activity[2]

       # print("\n{} from {} to {}".format(activity_name(label), datetime.datetime.fromtimestamp(start / 1000.0), datetime.datetime.fromtimestamp(end / 1000.0)))
        esense_trimmed = esense_data[np.logical_and(esense_data[:, 0] >= start,  esense_data[:, 0] <= end)]
        wrist_acc_trimmed = wrist_acc_data[np.logical_and(wrist_acc_data[:, 0] >= start,  wrist_acc_data[:, 0] <= end)]
        wrist_gyro_trimmed = wrist_gryo_data[np.logical_and(wrist_gryo_data[:, 0] >= start,  wrist_gryo_data[:, 0] <= end)]
        audio_trimmed = audio_data[np.logical_and(audio_data[:, 0] >= start,  audio_data[:, 0] <= end)]
        
        audio_fft = binned_audio_fft(audio_trimmed[:, 1], AUDIO_SAMPLE_RATE, AUDIO_MAX_FREQ, AUDIO_BINS)
        
        training_activities.append(Activity(label, 
                                            esense_trimmed,  
                                            wrist_acc_trimmed,
                                            wrist_gyro_trimmed,
                                            audio_fft[1]
                                            ))
    return training_activities


def load_activities(data_file):
    data = np.empty((0, 3))
    last_end_time = 0
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        is_start = True
        start_time = 0
        for row in csv_reader:
            if is_start:
                start_time = int(row[0])
                is_start = False
                if last_end_time != 0:
                    data = np.append(data, np.array([[ACTIVITIES["Other"], last_end_time, start_time]]), axis=0)
            else:
                try:
                    label = ACTIVITIES[row[2]]
                except KeyError:
                    label = ACTIVITIES["Other"]
                data = np.append(data, np.array([[label, start_time, int(row[0])]]), axis=0)
                last_end_time = int(row[0])
                is_start = True
    return data
            
def load_wrist(data_file):
    data = np.empty((0, 4))
    
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        last_sample = 0
        for row in csv_reader:
            if not row: 
                continue
            timestamp, _, acc_x, acc_y, acc_z = (float(row[0]), float(row[1]), float(row[2]), 
                                                     float(row[3]), float(row[4]))
            if (timestamp - last_sample >= DOWNSAMPLE_WRIST):
                data = np.append(data, np.array([[timestamp, acc_x, acc_y, acc_z]]), axis=0)
            last_sample = timestamp
    return data

def load_esense(data_file):
    data = np.empty((0, 7))
    
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
       # line_count = 0
       # start = 0
        last_sample = 0
       
        for row in csv_reader:
            if not row:
                continue
            timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = (float(row[0]), float(row[1]), float(row[2]), # Unpack Row
                                                                      float(row[3]), float(row[4]), float(row[5]), float(row[6]))
            
            if (timestamp - last_sample >= DOWNSAMPLE_ESENSE): # Downsample Esense Data
                data = np.append(data, np.array([[timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]]), axis=0)
                last_sample = timestamp
    
    return data

def load_audio(audio_file): 
        
   # with warnings.catch_warnings():
     #   warnings.filterwarnings("ignore", "C:\\Users\\aidan\\Anaconda3\\lib\\site-packages\\scipy\\io\\wavfile.py:273: WavFileWarning: Chunk (non-data) not understood, skipping it. WavFileWarning)")
    metadata = audio_metadata.load(audio_file)
    
    start = int(metadata.tags['TDRC'][0])
    #print ('Timestamp: {}'.format(start))

    fs, data = wavfile.read(audio_file)
    
    timestamps = np.flip(start - np.linspace(0, (data.shape[0] * 1000)/ fs, data.shape[0]))
    audio_data = np.column_stack((np.transpose(timestamps), data[:, 0]))
     
    return audio_data

def binned_audio_fft(data, fs, fmax, num_bins):
    T = 1.0 / fs
    N = data.shape[0]
    raw_fft = fft(data)
    x_fft = np.linspace(0.0, 1.0/(2.0*T), N//2)
    data_fft = 2.0/N * np.abs(raw_fft[0:N//2])
    
    bin_means, bin_edges, bin_number = binned_statistic(x_fft, data_fft, bins=num_bins, range=(0, 2000))
    #plt.plot(x_fft, data_fft)
    #plt.plot(bin_edges[:-1], bin_means)
    return (bin_edges, bin_means)



## sample spacing
#T = 1.0 / 8000.0
##x = np.linspace(0.0, N*T, N)
#y = data[:, 0]#np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#N = data.shape[0]
#
#print (N)
#yf = fft(y)
#xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#
#print (yf.shape)
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()
#
#data_fft = 2.0/N * np.abs(yf[0:N//2])
#bin_means, bin_edges, _ = binned_statistic(xf, data_fft, bins=100, range=(0, 2000))
#print (bin_means.shape)
#print (bin_edges[:-1].shape)
#plt.plot(bin_edges[:-1], bin_means)
    
    
def activity_name(label):
    for act, num in ACTIVITIES.items():
        if label == num:
            return act
    return "Other"

def plot_3axis(data, axis):
    x = np.linspace(1, data.shape[0], data.shape[0])
    
    plt.figure(1)
    plt.subplot(3, 1, 1)
    
    plt.plot(x, data[:, axis[0]], '.-')
    plt.title('X Axis')
    
    plt.subplot(3, 1, 2)
    plt.plot(x, data[:, axis[1]], '.-')
    plt.title('Y Axis')
    
    plt.subplot(3, 1, 3)
    plt.plot(x, data[:, axis[2]], '.-')
    plt.title('Z Axis')
    plt.xlabel('Samples')

if __name__=="__main__":
    #folder = os.getcwd() + '\\First_Data\\'
    training_data = []
    for f in os.walk(os.getcwd()):
        if ("Official_Data_Eating_Drinking_1" in f[0]):
            folder = f[0] + "\\"
            esense_data = load_esense(folder + ESENSE_FILE_NAME)
            wrist_acc_data = load_wrist(folder + ACCEL_FILE_NAME)
            wrist_gryo_data = load_wrist(folder + GYRO_FILE_NAME) 
            activities = load_activities(folder + MARKER_FILE_NAME)
            audio_data = load_audio(folder + AUDIO_FILE_NAME)
            training_data.extend(merge_sensor_data_stride(esense_data, wrist_acc_data, wrist_gryo_data, audio_data, activities))
            #raining_data.extend(merge_sensor_data(esense_data, wrist_acc_data, wrist_gryo_data, audio_data, activities))

           # plt.plot(abs(training_data[0].audio_data))
           # plot_3axis(wrist_acc_data, (1, 2, 3))
           # print (wrist_acc_data.shape[0])
           # print (1000 * wrist_acc_data.shape[0] / (wrist_acc_data[-1][0] - wrist_acc_data[0][0]))
            #plot_3axis(esense_data, (1, 2, 3))
            #print (esense_data.shape[0])
           # print (1000 * esense_data.shape[0] / (esense_data[-1][0] - esense_data[0][0]))
    #print (training_data)