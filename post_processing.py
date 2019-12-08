# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:08:53 2019

@author: Aidan
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz


ACTIVITIES = {
              "Other": 0,
              "Eating": 1,
              "Drinking": 2,
              "Smoking": 3,
              "Head Scratch": 4,
              "Walking": 5
              }

test_y = [0, 0, 1, 0, 0, 0, 2, 2, 2, 1, 2, 2, 0, 0, 0, 4, 3, 4, 4, 4, 
          0, 0, 0, 0, 5, 5, 5, 5, 5, 3, 3, 3, 5, 5, 5, 5, 0, 0, 0, 1, 
          4, 3, 1, 1, 1, 0, 0, 0, 4, 4, 0, 3, 3, 3, 2, 2, 1, 1, 1, 0]


def filter_by_min_length(data, min_length):
    for _ in range(min_length):
        data = remove_single_entries(data, True)
    return data

def remove_single_entries(raw_output, assume_prior=False):
    out = []
    
    for i, n in enumerate(raw_output):
        if (i == 0):
            out.append(n)
        elif (i == len(raw_output) - 1):
            out.append(n)
        else:
            if (not assume_prior):
                if (raw_output[i-1] == raw_output[i+1] and n != raw_output[i-1]):
                    out.append(raw_output[i-1])
                else:
                    out.append(n)
            else:
                if (raw_output[i] != raw_output[i-1] and raw_output[i] != raw_output[i+1]):
                    out.append(raw_output[i-1])
                else:
                    out.append(n)
                #if (raw_output )
    return (out)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lpf(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calculate_accuracy(truth, pred):
    diff = 0
    for i, v in enumerate(truth):
        if pred[i] != v:
            diff += 1
    return (len(truth) - diff) / float(len(truth))

def filter_bilateral(data, min_length):
    new_data = data.copy()
    for i in range(len(data)):
        y = data[i]
        i_l = i
        i_r = i
        while (i_l >= 0 and data[i_l] == y):
            i_l -= 1
        while (i_r < len(data) and data[i_r] == y):
            i_r += 1
       # print (i, ", ", i_r - i_l)
        if (i_r - i_l < min_length):
            #print ("\nChanged")
            new_data[i_l:i_r] = [y for _ in range(i_r-i_l)]

            
    return new_data

def display_results(data1, title1, data2, title2, data3=None, title3=None):
    
    numplots = 3 if data3.any() != None else 2
    
    x = np.linspace(0, len(data1)-1, len(data1))

    
    plt.subplot(numplots, 1, 1)
    plt.step(x, data1)
    
    y_ticks = list(ACTIVITIES.keys())
    y = [0, 1, 2, 3, 4, 5]
    plt.yticks(y, y_ticks)
    plt.title(title1)
    
    plt.subplot(numplots, 1, 2)  
    plt.step(x, data2)
    
    y_ticks = list(ACTIVITIES.keys())
    y = [0, 1, 2, 3, 4, 5]
    plt.yticks(y, y_ticks)
    plt.title(title2)
    
    if (data3.any() != None):
        plt.subplot(numplots, 1, 3)  
        plt.step(x, data3)
        
        y_ticks = list(ACTIVITIES.keys())
        y = [0, 1, 2, 3, 4, 5]
        plt.yticks(y, y_ticks)
        plt.title(title3)

    plt.show()
    
def display_results_2(data1, title1, data2, title2):
    
    numplots = 2
    
    x = np.linspace(0, len(data1)-1, len(data1))

    
    plt.subplot(numplots, 1, 1)
    plt.step(x, data1)
    
    y_ticks = list(ACTIVITIES.keys())
    y = [0, 1, 2, 3, 4, 5]
    plt.yticks(y, y_ticks)
    plt.title(title1)
    
    plt.subplot(numplots, 1, 2)  
    plt.step(x, data2)
    
    y_ticks = list(ACTIVITIES.keys())
    y = [0, 1, 2, 3, 4, 5]
    plt.yticks(y, y_ticks)
    plt.title(title2)
    
    plt.show()


if __name__== "__main__":
    
    y_pred = np.load("Prediction/y_pred.npy")
    y_true = np.load("Prediction/y_true.npy")
    print ("Initial Accuracy: ", str(calculate_accuracy(y_true, y_pred)))
    #out = filter_by_min_length(test_y, 3)
    #for i in range(50):
       # y_pred_smooth = filter_bilateral(y_pred, 50)#filter_by_min_length(y_pred, 5)#remove_single_entries(y_pred)
    y_pred_cont = butter_lpf(y_pred, .25, 10)
    y_pred_smooth = np.rint(y_pred_cont)
    #display_results_2(y_true,"Truth Data", y_pred, "Predicted")
    display_results(y_true,"Raw Data", y_pred_cont, "Continuous", y_pred_smooth, "Smoothed Data")
    
    print ("Single Smoothed Accuracy: ", str(calculate_accuracy(y_true, y_pred_smooth)))

