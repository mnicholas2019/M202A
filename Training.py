# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:22:33 2019

@author: Aidan
"""
import pandas as pd
from Features import mean

class Activity():
    
    def __init__(self, label, esense_data, wrist_acc_data, wrist_gyro_data, audio_data):
        self.label = label
        self.esense_data = esense_data
        self.wrist_acc_data = wrist_acc_data
        self.wrist_gyro_data = wrist_gyro_data
        self.audio_data = audio_data
    
    def calcFeaturesToABT(self, df, columns):
    	df2 = pd.DataFrame(data =[self.calculateFeatures()], columns = columns)
    	df = pd.concat([df, df2], ignore_index = True)
    	return df

    def calculateFeatures(self):
    	return [mean(self.esense_acc_x()) , mean(self.esense_acc_y()) , mean(self.esense_acc_z()) ,3, 4, self.label]
    	
    def esense_acc_x(self):
        return self.esense_data[:, 1]
    def esense_acc_y(self):
        return self.esense_data[:, 2]
    def esense_acc_z(self):
        return self.esense_data[:, 3]
    
    def esense_gyro_x(self):
        return self.esense_data[:, 4]
    def esense_gyro_y(self):
        return self.esense_data[:, 5]
    def esense_gyro_z(self):
        return self.esense_data[:, 6]
    
    def __str__(self):
        return "Training Data for {}".format(self.label)
    __repr__=__str__