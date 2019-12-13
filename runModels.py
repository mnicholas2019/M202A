from PreviousWork.existing_models import *
from PreviousWork.utils import *
from collections import Counter
import sys
import time
import pandas as pd
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
import numpy as np

testlist = ['w5_s100']
#feature_holdout = ['Mean', 'Stdev', 'Difference', 'Variance', 'Correlation', 'FFT', 'Real']
sensor_holdout = ['Esense_Accel', 'Esense_Gyro', 'Wrist_Accel', 'Wrist_Gyro', 'Just_Wrist', 'Just_Esense']
for testname in testlist:
	epochs = 40
	d_name = 'M202A Project'
	num_classes, sensors, locations, label_names, f_hz, dimensions, path = get_details('M202A Project')
	path = './DataVariationOther/' + testname + '/'


	print('testname:',testname)


	# for index, feat in enumerate(feature_holdout):
	# 	x_train0, x_val0, x_test0, y_train_binary, y_val_binary, y_test_binary = load_dataset(d_name, path, num_classes)
	# 	print('shape', x_train0.shape)
	# 	if feat == 'JustFFT':
	# 	    for i in range(8):
	# 	    	x_train0 = np.delete(x_train0[:][:], 0, axis = 2)
	# 	    	x_val0 = np.delete(x_val0[:][:], 0, axis = 2)
	# 	    	x_test0 = np.delete(x_test0[:][:], 0, axis = 2)
	# 	elif feat != 'FFT' and feat != 'Real':
	# 		x_train0 = np.delete(x_train0[:][:], index, axis = 2)
	# 		x_val0 = np.delete(x_val0[:][:], index, axis = 2)
	# 		x_test0 = np.delete(x_test0[:][:], index, axis = 2)
	# 	elif feat == 'FFT':
	# 		x_train0 = np.delete(x_train0[:][:], index+2, axis = 2)
	# 		x_val0 = np.delete(x_val0[:][:], index+2, axis = 2)
	# 		x_test0 = np.delete(x_test0[:][:], index+2, axis = 2)
	# 		x_train0 = np.delete(x_train0[:][:], index+1, axis = 2)
	# 		x_val0 = np.delete(x_val0[:][:], index+1, axis = 2)
	# 		x_test0 = np.delete(x_test0[:][:], index+1, axis = 2)
	# 		x_train0 = np.delete(x_train0[:][:], index, axis = 2)
	# 		x_val0 = np.delete(x_val0[:][:], index, axis = 2)
	# 		x_test0 = np.delete(x_test0[:][:], index, axis = 2)

		# print('Doing', feat, 'Test: ', x_train0.shape)
	for index, sensor in enumerate(sensor_holdout):
		x_train0, x_val0, x_test0, y_train_binary, y_val_binary, y_test_binary = load_dataset(d_name, path, num_classes)
		print('shape', x_train0.shape)
		if index < 4:
			for i in range(3):
				x_train0 = np.delete(x_train0[:][:], index*3, axis = 1)
				x_val0 = np.delete(x_val0[:][:], index*3, axis = 1)
				x_test0 = np.delete(x_test0[:][:], index*3, axis = 1)
		elif index == 4:
			for i in range(6):
				x_train0 = np.delete(x_train0[:][:], 0, axis = 1)
				x_val0 = np.delete(x_val0[:][:], 0, axis = 1)
				x_test0 = np.delete(x_test0[:][:], 0, axis = 1)
		else:
			for i in range(6):
				x_train0 = np.delete(x_train0[:][:], 6, axis = 1)
				x_val0 = np.delete(x_val0[:][:], 6, axis = 1)
				x_test0 = np.delete(x_test0[:][:], 6, axis = 1)


		print('Doing', sensor, 'Test: ', x_train0.shape)

			

		num_classes = 6 #only for no other


		network_type = 'M202A_CNN'
		X_train, X_val, X_test = reshape_data(x_train0, x_val0, x_test0, network_type)


		batch_size = 256
		_, num_streams, num_features = x_train0.shape

		model = model_CNN(num_streams, num_features, num_classes, num_feat_map=32, p=0.3)
		#print(model.summary())

		print('model training ...')
		print('num epochs', epochs)
		model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
		              optimizer='adam',
		              metrics=['accuracy'])

		model_dir = f'Models/{d_name}'

		name = 'CNN_{}'.format(int(time.time()))
		tensorboard = TensorBoard(log_dir = 'logs/{}'.format(name))

		if not os.path.exists(model_dir):
		    os.makedirs(model_dir)

		# checkpoint
		filepath= f"best_{name}.hdf5"
		chk_path = os.path.join(model_dir, filepath)
		checkpoint = ModelCheckpoint(chk_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

		model.fit(X_train, y_train_binary,
		          batch_size=300,
		          epochs=epochs,
		          verbose=1,
		          shuffle=True,
		          validation_data=(X_val, y_val_binary),
		          callbacks=[tensorboard, checkpoint])

		model.save('Sensor_Holdout/' + sensor +  '/mfccmodel.hdf5')

		model = load_model(chk_path)

		y_pred = np.argmax(model.predict(X_test), axis=1)
		y_true = np.argmax(y_test_binary, axis=1)
		print(y_pred, y_true)
		cf_matrix = confusion_matrix(y_true, y_pred)

		np.save("Sensor_Holdout/" + sensor +  "/prediction", y_pred)
		np.save("Sensor_Holdout/" + sensor + "/truth", y_true)

		print(cf_matrix)
		class_wise_f1 = f1_score(y_true, y_pred, average=None)
		print('the mean-f1 score: {:.4f}'.format(np.mean(class_wise_f1)))
		accuracy = accuracy_score(y_true, y_pred)
		print('accuracy is: {:.4f}'.format(accuracy))





