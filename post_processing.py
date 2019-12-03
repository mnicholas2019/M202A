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

def smooth_output(raw_output):
    return (raw_output)

def display_results(actual, predicted=None):
    y_ticks = list(ACTIVITIES.keys())
    
    x = np.linspace(0, len(actual)-1, len(actual))
    y = [0, 1, 2, 3, 4, 5]
    plt.step(x, actual)
    plt.yticks(y, y_ticks)
    
    if (predicted is not None):
        plt.step(x, predicted)
    
    plt.show()


if __name__== "__main__":
    display_results(test_y)
    print(smooth_output(test_y))