import numpy as np 
import os
import pandas as pd
#from PreviousWork.utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

folder = 'Sensor_Holdout'
windows_strides = ['Esense_Accel', 'Esense_Gyro', 'Just_Wrist', 'Wrist_Accel', 'Wrist_Gyro', 'Just_Esense']


print('\n')
for ws in windows_strides:
	path = folder + os.path.sep + ws + os.path.sep
	pred = np.load(path + 'prediction.npy')
	truth = np.load(path + 'truth.npy')

	print(ws)
	print("number of labels:", len(pred))
	class_wise_f1 = f1_score(truth, pred, average=None)
	accuracy = accuracy_score(truth, pred)
	cf_matrix = confusion_matrix(truth, pred)
	#cf_matrix, accuracy, micro_f1, macro_f1 = calculate_metrics()
	print(np.mean(class_wise_f1), accuracy)
	print(cf_matrix)
	print('\n')
