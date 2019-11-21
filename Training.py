# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:22:33 2019

@author: Aidan
"""
import pandas as pd
from Features import *

class Activity():
    
    def __init__(self, label, esense_data, wrist_acc_data, wrist_gyro_data, audio_data):
        self.label = label
        self.esense_data = esense_data
        self.wrist_acc_data = wrist_acc_data
        self.wrist_gyro_data = wrist_gyro_data
        self.audio_data = audio_data
    
    def calcFeaturesToABT(self, df, columns, toPrint):
    	df2 = pd.DataFrame(data =[self.calculateFeatures(toPrint)], columns = columns)
    	df = pd.concat([df, df2], ignore_index = True)
    	return df

    def calculateFeatures(self, toPrint):
        data_streams = [self.esense_acc_x(), self.esense_acc_y(), self.esense_acc_z(), 
                        self.esense_gyro_x(), self.esense_gyro_y(), self.esense_gyro_z(),
                        self.wrist_acc_x(), self.wrist_acc_y(), self.wrist_acc_z(),
                        self.wrist_gyro_x(), self.wrist_gyro_y(), self.wrist_gyro_z()]
        features = []
        for d_stream in data_streams:
            features.append(mean(d_stream))
            features.append(stdev(d_stream))
            features.append(range(d_stream))
            features.append(variance(d_stream))
        features = features + correlations(data_streams[0], data_streams[1], data_streams[2])
        features = features + correlations(data_streams[3], data_streams[4], data_streams[5])
        features = features + correlations(data_streams[6], data_streams[7], data_streams[8])
        features = features + correlations(data_streams[9], data_streams[10], data_streams[11])
        if toPrint == 0:
            print('features:',features)

        return features
        #return [mean(self.esense_acc_x()) , mean(self.esense_acc_y()) , mean(self.esense_acc_z()) ,3, 4, self.label]


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

    def wrist_acc_x(self):
        return self.wrist_acc_data[:, 1]
    def wrist_acc_y(self):
        return self.wrist_acc_data[:, 2]
    def wrist_acc_z(self):
        return self.wrist_acc_data[:, 3]

    def wrist_gyro_x(self):
        return self.wrist_gyro_data[:, 1]
    def wrist_gyro_y(self):
        return self.wrist_gyro_data[:, 2]
    def wrist_gyro_z(self):
        return self.wrist_gyro_data[:, 3]
    
    def __str__(self):
        return "Training Data for {}".format(self.label)
    __repr__=__str__