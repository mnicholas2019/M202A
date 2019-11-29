import numpy as np
from sklearn.model_selection import train_test_split


target = np.load('model_data/targetNP.npy')
data = np.load('model_data/dataframeNP.npy')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
