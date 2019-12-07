import numpy as np


data = np.load("model_data/dataframeNP.npy")
target = np.load("model_data/targetNP.npy")

print(type(data))
print(data[0][0][4])
print(type(data[0][0][4]))
print(target[0][0])
print(type(target[0][0]))
print(data.shape)
print(target.shape)


print("\n\n\n")

data1 = np.load("modelfft_data/dataframeNP.npy")
target1 = np.load("modelfft_data/targetNP.npy")

print(type(data1))
print(data1[0][0][7])
print(type(data1[0][0][7]))
print(target1[0][0])
print(type(target1[0][0]))
print(data1.shape)
print(target1.shape)

data2 = np.load("model_test_data/dataframeNP.npy")
target2 = np.load("model_test_data/targetNP.npy")

print("\n\n\n")
print(type(data2))
print(data2[0][0][7])
print(type(data2[0][0][7]))
print(target2[0][0])
print(type(target2[0][0]))
print(data2.shape)
print(target2.shape)

#print("\n\nEqual:", data1==data2)