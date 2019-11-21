# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:13:14 2019

@author: aidan
"""
import numpy as np

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