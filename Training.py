# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:22:33 2019

@author: Aidan
"""
import pandas as pd
from Features import *
import numpy as np

ESENSE_SAMPLE_RATE = 10 # Esense Sampling Rate (Approx)
WRIST_SAMPLE_RATE = 20 # Wrist Sampling Rate (Approx)
IMU_BINS = 10 # Number of Frequency bins for fft audio

class Activity():
    
    def __init__(self, label, esense_data, wrist_acc_data, wrist_gyro_data, audio_data):
        self.label = label
        self.esense_data = esense_data
        self.wrist_acc_data = wrist_acc_data
        self.wrist_gyro_data = wrist_gyro_data
        self.audio_data = audio_data
        #self.audio_data = audio_data
    
    def calcFeaturesToABT(self, df, columns):
    	df2 = pd.DataFrame(data =[self.calculateFeatures(ABT=1)], columns = columns)
    	df = pd.concat([df, df2], ignore_index = True)
    	return df

    def calculateFeatures(self, ABT):
        if ABT:
            data_streams = [self.esense_acc_x(), self.esense_acc_y(), self.esense_acc_z(), 
                            self.esense_gyro_x(), self.esense_gyro_y(), self.esense_gyro_z(),
                            self.wrist_acc_x(), self.wrist_acc_y(), self.wrist_acc_z(),
                            self.wrist_gyro_x(), self.wrist_gyro_y(), self.wrist_gyro_z()]
            features = []
            for d_stream in data_streams:
                features.append(mean(d_stream))
                features.append(stdev(d_stream))
                features.append(difference(d_stream))
                features.append(variance(d_stream))
            features = features + correlations(data_streams[0], data_streams[1], data_streams[2])
            features = features + correlations(data_streams[3], data_streams[4], data_streams[5])
            features = features + correlations(data_streams[6], data_streams[7], data_streams[8])
            features = features + correlations(data_streams[9], data_streams[10], data_streams[11])
            features.append(self.label)

            return features
        else:
            data_streams = [self.esense_acc_x(), self.esense_acc_y(), self.esense_acc_z(), 
                            self.esense_gyro_x(), self.esense_gyro_y(), self.esense_gyro_z(),
                            self.wrist_acc_x(), self.wrist_acc_y(), self.wrist_acc_z(),
                            self.wrist_gyro_x(), self.wrist_gyro_y(), self.wrist_gyro_z()]
            features = ['mean', 'stdev', 'range', 'variance']
            window_calc = []
            for i, d_stream in enumerate(data_streams):

                stream_data = []
                stream_data.append(mean(d_stream))
                stream_data.append(stdev(d_stream))
                stream_data.append(difference(d_stream))
                stream_data.append(variance(d_stream))
                if i < 3:
                    corr = correlations(data_streams[0], data_streams[1], data_streams[2])
                elif i < 6:
                    corr = correlations(data_streams[3], data_streams[4], data_streams[5])
                elif i < 9:
                    corr = correlations(data_streams[6], data_streams[7], data_streams[8])
                else:
                    corr = correlations(data_streams[9], data_streams[10], data_streams[11])
                if i%3 == 0:
                    stream_data.append(corr[0])
                elif i%3 == 1:
                    stream_data.append(corr[2])
                else:
                    stream_data.append(corr[1])

                window_calc.append(stream_data)
    
            # Returns esense fft in form of a list(acc_x, acc_y, acc_z). Other lists are the same format
            fft = []
            esense_acc_fft = imu_fft(self.esense_acc(), ESENSE_SAMPLE_RATE, IMU_BINS)
            esense_gyro_fft = imu_fft(self.esense_gyro(), ESENSE_SAMPLE_RATE, IMU_BINS)
            wrist_acc_fft = imu_fft(self.wrist_acc(), WRIST_SAMPLE_RATE, IMU_BINS)
            wrist_gyro_fft = imu_fft(self.wrist_gyro(), WRIST_SAMPLE_RATE, IMU_BINS)
            

            fft = [esense_acc_fft[0][0:3].tolist(), esense_acc_fft[1][0:3].tolist(), esense_acc_fft[2][0:3].tolist(),
                   esense_gyro_fft[0][0:3].tolist(), esense_gyro_fft[1][0:3].tolist(), esense_gyro_fft[2][0:3].tolist(),
                   wrist_acc_fft[0][0:3].tolist(), wrist_acc_fft[1][0:3].tolist(), wrist_acc_fft[2][0:3].tolist(),
                   wrist_gyro_fft[0][0:3].tolist(), wrist_gyro_fft[1][0:3].tolist(), wrist_gyro_fft[2][0:3].tolist()]

            #print("\n\nFFT:\n ", fft)
            #print("\n\nwindow: ", window_calc)

            tester = np.concatenate((window_calc, fft), axis = 1)
            #print ("\n\ntester", tester)
            #print("\n\nshape: ", tester.shape)
            
            mfcc = mfcc_audio(self.audio_data[:, 1])

            return tester
        
        # Returns 12x16 mfcc (12 coefficients, 512ms windows (hop_length)). (WINDOW_LENGTH) * SAMPLING_FREQUENCY / HOP_LENGTH = columns of mfcc
        #
        #print("we are here", mfcc)
        


    def esense_acc(self):
        return self.esense_data[:, 1:4]
    def esense_gyro(self):
        return self.esense_data[:, 4:]
    def wrist_acc(self):
        return self.wrist_acc_data[:, 1:]
    def wrist_gyro(self):
        return self.wrist_gyro_data[:, 1:]

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