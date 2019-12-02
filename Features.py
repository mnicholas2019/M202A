# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:13:14 2019

@author: aidan
"""
import numpy as np
import librosa
from scipy.fftpack import fft
from scipy.stats import binned_statistic

MFCC_NUM = 12
AUDIO_SAMPLE_RATE = 8000 

def mean(data):
    return np.mean(data)

def stdev(data):
    return np.std(data)

def range(data):
    return np.max(data) - np.min(data)

def variance(data):
    return np.var(data)

def correlations(data1, data2, data3):
    coef = np.corrcoef([data1,data2,data3])
    return [coef[1,0], coef[2,0], coef[1,2]]


# FFT of 3-Axis Sensor. Use Nyquist Sampling Theorem so fmax = 2*fs
def imu_fft(data, fs, num_bins):
    imu_fft_out = (binned_fft(data[:, 0], fs, 2*fs, num_bins)[1], 
                  (binned_fft(data[:, 1], fs, 2*fs, num_bins)[1], 
                  (binned_fft(data[:, 2], fs, 2*fs, num_bins)[1])))
        
    return imu_fft_out

def mfcc_audio(audio_data):
    mfcc = librosa.feature.mfcc(audio_data, n_mfcc=MFCC_NUM, sr=AUDIO_SAMPLE_RATE)
    return mfcc
                  

def binned_fft(data, fs, fmax, num_bins):
    T = 1.0 / fs
    N = data.shape[0]
    raw_fft = fft(data)
    x_fft = np.linspace(0.0, 1.0/(2.0*T), N//2)
    data_fft = 2.0/N * np.abs(raw_fft[0:N//2])
    
    bin_means, bin_edges, bin_number = binned_statistic(x_fft, data_fft, bins=num_bins, range=(0, fmax))
    #plt.plot(x_fft, data_fft)
    #plt.plot(bin_edges[:-1], bin_means)
    return (bin_edges, bin_means) 