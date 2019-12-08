import numpy as np 

no_other_train = np.load("noOtherData/dataframeTrainingNP.npy")
y_no_other = np.load("noOtherData/targetTrainingNP.npy")
y_no_other = np.load("noOtherData/targetTestNP.npy")

x_train = np.load("modelfft_data/dataframeTrainingNP.npy")
y_train = np.load("modelfft_data/targetTrainingNP.npy")

c = 0
for val in y_no_other:
	if val[1] == 1:
		c = c+1
print(c)

print(no_other_train.shape)
print(y_no_other.shape)
print(x_train.shape)
print(y_train.shape)

