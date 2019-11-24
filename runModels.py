from PreviousWork.existing_models import *
from PreviousWork.utils import *
from collections import Counter
import sys
import time
import pandas as pd
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model


d_name = 'M202A Project'
num_classes, sensors, locations, label_names, f_hz, dimensions, path = get_details('M202A Project')

x_train0, x_val0, x_test0, y_train, y_val, y_test = load_dataset(d_name, path, num_classes)

x_train0 = x_train0.to_numpy()
x_val0 = x_val0.to_numpy()
x_test0 = x_test0.to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()
print(x_train0.shape)
print(y_train.shape)

x_train = np.reshape(x_train0, (1514,60,1))
x_val = np.reshape(x_val0, (1514,60,1))
x_test = np.reshape(x_test0, (1514,60,1))
y_train_real = np.zeros((1514,5))
for i, label in enumerate(y_train):
	y_train_real[i][label] = 1
	if label != 1 and label != 2:
		print("no other!!! ", label)

print('\n\n\nhere: ', y_train_real)
y_train = y_train_real
y_val = y_train_real
y_test = y_train_real

network_type = 'CNN'
win_len, dim = 1, 60

model = model_CNN(dim, win_len, num_classes, num_feat_map=32, p=0.3)
print(model.summary())


print('model training ...')
epochs = 20
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model_dir = f'Models/{d_name}'


name = '{}_d{}_bn{}_{}'.format(network_type, 'cnn', 'bnn', int(time.time()))
tensorboard = TensorBoard(log_dir = './Models/M202A Project/{}'.format(name))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# checkpoint
filepath= f"{name}.hdf5"
chk_path = os.path.join(model_dir, filepath)
checkpoint = ModelCheckpoint(chk_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(x_train, y_train,
          batch_size=300,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(x_val, y_val),
          callbacks=[tensorboard, checkpoint])


model = load_model(chk_path)


y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_binary, axis=1)
cf_matrix = confusion_matrix(y_true, y_pred)
print(cf_matrix)
class_wise_f1 = f1_score(y_true, y_pred, average=None)
print('the mean-f1 score: {:.4f}'.format(np.mean(class_wise_f1)))
accuracy = accuracy_score(y_true, y_pred)
print('accuracy is: {:.4f}'.format(accuracy))


