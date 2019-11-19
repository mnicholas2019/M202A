# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:22:33 2019

@author: Aidan
"""

import pandas as pd
class Activity():
    
    def __init__(self, label, esense_data, wrist_acc_data, wrist_gyro_data, audio_data):
        self.label = label
        self.esense_data = esense_data
        self.wrist_acc_data = wrist_acc_data
        self.wrist_gyro_data = wrist_gyro_data
        self.audio_data = audio_data
    
    def __str__(self):
        return "Training Data for {}".format(self.label)
    __repr__=__str__