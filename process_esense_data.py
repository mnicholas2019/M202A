# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:22:33 2019

@author: Aidan
"""

import csv
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from Training import Activity

ACCEL_FILE_NAME = "ACCELEROMETER--org.md2k.motionsense--MOTION_SENSE_HRV--RIGHT_WRIST.csv"
GYRO_FILE_NAME = "GYROSCOPE--org.md2k.motionsense--MOTION_SENSE_HRV--RIGHT_WRIST.csv"
MARKER_FILE_NAME = "MARKER--org.md2k.mcerebrum--PHONE.csv"
ESENSE_FILE_NAME = "esense_data.txt"

plt.style.use('seaborn-whitegrid')

ACTIVITIES = {
              "Other": 0,
              "Eating": 1,
              "Drinking": 2,
              "Smoking": 3,
              "Head scratch": 4
              }

def merge_sensor_data(esense_data, wrist_acc_data, wrist_gryo_data, audio_data, activity_data):
    training_activities = []    
    for activity in activity_data:
        label = int(activity[0])
        start = activity[1]
        end = activity[2]

        print("\n{} from {} to {}".format(activity_name(label), datetime.datetime.fromtimestamp(start / 1000.0), 
                                                                datetime.datetime.fromtimestamp(end / 1000.0)))
        esense_trimmed = esense_data[np.logical_and(esense_data[:, 0] >= start,  esense_data[:, 0] <= end)]
        wrist_acc_trimmed = wrist_acc_data[np.logical_and(wrist_acc_data[:, 0] >= start,  wrist_acc_data[:, 0] <= end)]
        wrist_gyro_trimmed = wrist_gryo_data[np.logical_and(wrist_gryo_data[:, 0] >= start,  wrist_gryo_data[:, 0] <= end)]
        audio_trimmed = audio_data#audio_data[np.logical_and(audio_data[:, 0] >= start,  audio_data[:, 0] <= end)]
        
        training_activities.append(Activity(label, 
                                            esense_trimmed,  
                                            wrist_acc_trimmed,
                                            wrist_gyro_trimmed,
                                            audio_trimmed
                                            ))
    return training_activities


def load_activities(data_file):
    data = np.empty((0, 3))
    
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        is_start = True
        start_time = 0
        for row in csv_reader:
            if is_start:
                start_time = int(row[0])
                is_start = False
            else:
                try:
                    label = ACTIVITIES[row[2]]
                except KeyError:
                    label = "Other"
                data = np.append(data, np.array([[label, start_time, int(row[0])]]), axis=0)
                is_start = True
    return data
            
def load_wrist(data_file):
    data = np.empty((0, 4))
    
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if not row: 
                continue
            timestamp, _, acc_x, acc_y, acc_z = (float(row[0]), float(row[1]), float(row[2]), 
                                                     float(row[3]), float(row[4]))
            data = np.append(data, np.array([[timestamp, acc_x, acc_y, acc_z]]), axis=0)
    return data

def load_esense(data_file):
    data = np.empty((0, 7))
    
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
       # line_count = 0
       # start = 0
        for row in csv_reader:
            if not row:
                continue
            timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = (float(row[0]), float(row[1]), float(row[2]), # Unpack Row
                                                                      float(row[3]), float(row[4]), float(row[5]), float(row[6]))
    
            #sample_time = datetime.datetime.fromtimestamp(timestamp / 1000.0)#.strftime('%c') # Convert Timestamp to Local Time
            #if (line_count == 0):
             #   start = sample_time
    
            #offset_time = (sample_time - start).total_seconds() # Seconds since first sample
    
            data = np.append(data, np.array([[timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]]), axis=0)
           # line_count += 1
    
    return data

   # print ("Sampling Rate: {}".format(data.shape[0] / (sample_time - start).total_seconds()))
    
    
#    x = data[:, 0]
#    sample_rate = 15
#    num_samples = int(sample_rate * (sample_time - start).total_seconds() + .5)
#    print (num_samples)
#    acc_x_downsampled = signal.resample(data[:, 1], num_samples)
#    
#    plt.figure(1)
#    x_downsampled = signal.resample(x, num_samples)
#    
#    plt.subplot(2, 1, 1)
#    plt.plot(x, data[:, 1], '.-')
#    plt.subplot(2, 1, 2)
#    plt.plot(x_downsampled, acc_x_downsampled, '.-')
#    plt.show()

    ##
    ### Plot Accelerometer/Gyroscope Data
    ##x = data[:, 0]
    ##
    ##plt.figure(0)
    ##plt.subplot(3, 1, 1)
    ##
    ##plt.plot(x, data[:, 1], '.-')
    ##plt.title('Accelerometer: X')
    ##
    ##plt.subplot(3, 1, 2)
    ##plt.plot(x, data[:, 2], '.-')
    ##plt.title('Accelerometer: Y')
    ##
    ##plt.subplot(3, 1, 3)
    ##plt.plot(x, data[:, 3], '.-')
    ##plt.title('Accelerometer: Z')
    ##plt.xlabel('Seconds (s)')
    
    
    ##plt.figure(1)
    ##plt.subplot(3, 1, 1)
    ##
    ##plt.plot(x, data[:, 4], '.-')
    ##plt.title('Gyroscope: X')
    ##
    ##plt.subplot(3, 1, 2)
    ##plt.plot(x, data[:, 5], '.-')
    ##plt.title('Gyroscope: Y')
    ##
    ##plt.subplot(3, 1, 3)
    ##plt.plot(x, data[:, 6], '.-')
    ##plt.title('Gyroscope: Z')
    ##plt.xlabel('Seconds (s)')
    ##
    ##plt.show()
def activity_name(label):
    for act, num in ACTIVITIES.items():
        if label == num:
            return act
    return "Other"

if __name__=="__main__":
    #folder = os.getcwd() + '\\First_Data\\'
    folder = os.getcwd() + '/First_Data/'

    esense_data = load_esense(folder + ESENSE_FILE_NAME)
    wrist_acc_data = load_wrist(folder + ACCEL_FILE_NAME)
    wrist_gryo_data = load_wrist(folder + GYRO_FILE_NAME) 
    activities = load_activities(folder + MARKER_FILE_NAME)
    audio_data = ""
    training_data = merge_sensor_data(esense_data, wrist_acc_data, wrist_gryo_data, audio_data, activities)
    print (training_data)