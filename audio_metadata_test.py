# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:59:17 2019

@author: Aidan
"""

import audio_metadata
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.stats import binned_statistic

audio_file = 'C:\\Users\\Aidan\\Documents\\git_repos\\M202A\\Official_Data_Eating_Drinking_1\\audio.wav'

metadata = audio_metadata.load(audio_file)

start = int(metadata.tags['TDRC'][0])
print ('Timestamp: {}'.format(start))

fs, data = wavfile.read(audio_file)

ts = 1000.0/fs # Sample Period in millis

timestamps = np.flip(start - np.linspace(0, (data.shape[0] * 1000)/ fs, data.shape[0]))
audio_data = np.column_stack((np.transpose(timestamps), data[:, 0]))



# sample spacing
T = 1.0 / 8000.0
#x = np.linspace(0.0, N*T, N)
y = data[:, 0]#np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
N = data.shape[0]

print (N)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

print (yf.shape)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

data_fft = 2.0/N * np.abs(yf[0:N//2])
bin_means, bin_edges, _ = binned_statistic(xf, data_fft, bins=100, range=(0, 2000))
print (bin_means.shape)
print (bin_edges[:-1].shape)
plt.plot(bin_edges[:-1], bin_means)

#x = timestamps
#plt.subplot(2, 1, 1)
#plt.plot(x, data[:, 0], '.-')
#plt.subplot(2, 1, 2)
#plt.plot(x, data[:, 1], '.-')
#plt.show()