# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:08:53 2019

@author: Aidan
"""
import numpy as np
import matplotlib.pyplot as plt

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

def display_results(data1, title1, data2, title2):
    
    x = np.linspace(0, len(data1)-1, len(data1))

    plt.subplot(2, 1, 1)
    plt.step(x, data1)
    
    y_ticks = list(ACTIVITIES.keys())
    y = [0, 1, 2, 3, 4, 5]
    plt.yticks(y, y_ticks)
    plt.title(title1)
    
    plt.subplot(2, 1, 2)  
    plt.step(x, data2)
    
    y_ticks = list(ACTIVITIES.keys())
    y = [0, 1, 2, 3, 4, 5]
    plt.yticks(y, y_ticks)
    plt.title(title2)

    plt.show()


if __name__== "__main__":
    out = filter_by_min_length(test_y, 3)
   # out = remove_single_entries(test_y)
    display_results(test_y,"Raw Data", out, "Smoothed Data")
