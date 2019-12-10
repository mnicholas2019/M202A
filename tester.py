import numpy as np 

labels = np.load('DataVariationOther/w1_s500/targetTestNP.npy')
for lab in labels:
	print(lab)