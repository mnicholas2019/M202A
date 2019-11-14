# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:59:17 2019

@author: Aidan
"""

import audio_metadata

metadata = audio_metadata.load('test_audio.wav')

print (metadata)

print ('Timestamp: {}'.format(int(metadata.tags['TDRC'][0])))