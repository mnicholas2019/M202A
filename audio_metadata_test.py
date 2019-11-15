# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:59:17 2019

@author: Aidan
"""

import audio_metadata
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

audio_file = 'C:\\Users\\Aidan\\Documents\\git_repos\M202A\\First_Data\\audio.wav'

metadata = audio_metadata.load(audio_file)

print (metadata)
start = int(metadata.tags['TDRC'][0])
print ('Timestamp: {}'.format(start))

fs, data = wavfile.read(audio_file)

ts = 1000.0/fs # Sample Period in millis

timestamps = np.linspace(0, (data.shape[0] * 1000)/ fs, data.shape[0]) + start
audio_data = np.column_stack((np.transpose(timestamps), data[:, 0]))

x = np.linspace(1, data.shape[0], data.shape[0])
plt.subplot(2, 1, 1)
plt.plot(x, data[:, 0], '.-')
plt.subplot(2, 1, 2)
plt.plot(x, data[:, 1], '.-')
plt.show()