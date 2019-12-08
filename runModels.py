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



d_name = 'M202A Project'
num_classes, sensors, locations, label_names, f_hz, dimensions, path = get_details('M202A Project')
x_train0, x_val0, x_test0, y_train_binary, y_val_binary, y_test_binary = load_dataset(d_name, path, num_classes)

num_classes = 6 #only for no other


network_type = 'M202A_CNN'
X_train, X_val, X_test = reshape_data(x_train0, x_val0, x_test0, network_type)

batch_size = 256
_, num_streams, num_features = x_train0.shape

model = model_CNN(num_streams, num_features, num_classes, num_feat_map=32, p=0.3)
print(model.summary())

print('model training ...')
epochs = 40
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

model.save(f'final_{name}.hdf5')

model = load_model(chk_path)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_binary, axis=1)
print(y_pred, y_true)
cf_matrix = confusion_matrix(y_true, y_pred)

np.save("Prediction/y_pred_with_val_noMFCC.npy", y_pred)
np.save("Prediction/y_true_with_val_noMFCC.npy", y_true)

print(cf_matrix)
class_wise_f1 = f1_score(y_true, y_pred, average=None)
print('the mean-f1 score: {:.4f}'.format(np.mean(class_wise_f1)))
accuracy = accuracy_score(y_true, y_pred)
print('accuracy is: {:.4f}'.format(accuracy))





