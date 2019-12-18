import numpy as np
import random
from osgeo import gdal
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.layers import Bidirectional
from keras.utils import plot_model
from keras.layers import Dense,Dropout,Activation,Convolution3D,MaxPooling3D,Flatten
from keras.optimizers import Adam
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K
from time import time

# Read land use classification layer data
file_land2000 = '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Landuse_final/lan2000_final.tif'
data_land2000 = gdal.Open(file_land2000)

im_height = data_land2000.RasterYSize  # Number of rows in the raster matrix
im_width = data_land2000.RasterXSize  # Number of columns in the raster matrix

# Land use data
im_data_land2000 = data_land2000.ReadAsArray(0, 0, im_width, im_height)

number = 0
# Number of pixels in Shanghai
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
            if im_data_land2000[row][col] != 0 :
                number = number + 1

print("number of ShangHai:\n",number)

# load data
data_sample_2003 = np.load("/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/CNN_features/CNN_data_sample/CNN_35_35/data_sample_2003_npy.npy").reshape(-1,35,35,1)
data_sample_2004 = np.load("/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/CNN_features/CNN_data_sample/CNN_35_35/data_sample_2004_npy.npy").reshape(-1,35,35,1)
data_sample_2005 = np.load("/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/CNN_features/CNN_data_sample/CNN_35_35/data_sample_2005_npy.npy").reshape(-1,35,35,1)

data_label_2005 = np.loadtxt('/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Labels/data_label_2009.txt')

# concatenate
data_sample_03to05 = np.concatenate((data_sample_2003,data_sample_2004,data_sample_2005), axis=3)
data_sample_label_03to05 = data_label_2005

# get training data
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []

    for i in range(length):
        random_list.append(random.randint(start, stop))
    return np.array(random_list)

num_of_year = int(number * 0.2)
random_list = random_int_list(0, number - 1, num_of_year)

cnn_data_03to05 = np.zeros((num_of_year, 35, 35, 3))
cnn_data_label_03to05 = np.zeros((num_of_year, 1))

for i in range(0, num_of_year):
    temp = random_list[i]
    cnn_data_03to05[i] = data_sample_03to05[temp]
    cnn_data_label_03to05[i] = data_sample_label_03to05[temp]

cnn_data = cnn_data_03to05
cnn_data_label = cnn_data_label_03to05

# cnn_data was used to train
train_num = int(cnn_data.shape[0] * 0.7)  # set number of training samples
test_num = cnn_data.shape[0] - train_num  # set number of test samples

# get final training data from cnn_data
cnn_data_train = np.zeros((train_num, 35, 35, 3))
cnn_data_test = np.zeros((test_num, 35, 35, 3))
cnn_data_train_label = np.zeros((train_num, 1))
cnn_data_test_label = np.zeros((test_num, 1))
for i in range(train_num):
    cnn_data_train[i] = cnn_data[i]
    cnn_data_train_label[i] = cnn_data_label[i]
for j in range(train_num, cnn_data.shape[0]):
    cnn_data_test[j - train_num] = cnn_data[j]
    cnn_data_test_label[j - train_num] = cnn_data_label[j]

# transform label into one-hot
cnn_data_train_label = np_utils.to_categorical(cnn_data_train_label, num_classes=7)
cnn_data_test_label = np_utils.to_categorical(cnn_data_test_label, num_classes=7)

cnn_data_train = np.reshape(cnn_data_train,(cnn_data_train.shape[0],cnn_data_train.shape[1],cnn_data_train.shape[2],cnn_data_train.shape[3],1))
cnn_data_test = np.reshape(cnn_data_test,(cnn_data_test.shape[0],cnn_data_test.shape[1],cnn_data_test.shape[2],cnn_data_test.shape[3],1))

t1 = time()
# set up a 3D CNN model
model=Sequential()
model.add(Conv3D(32, (3, 3, 3), input_shape=(35, 35, 3, 1), padding="same") )
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1),  padding="same"))

# This is adding a second layer of neural network, convolution layer, excitation function, pooling layer
model.add(Conv3D(64, (3, 3, 3), input_shape=(35, 35, 3, 1), padding="same") )
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),  padding="same"))

# add a third convolution layer
model.add(Conv3D(32, (3, 3, 3), input_shape=(35, 35, 3, 1), padding="same") )
model.add(Activation('relu'))

# sort into one dimension
model.add(Flatten())

model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

# define your optimize
rms = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

# adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.6, amsgrad=False)
# model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])

print('\nTraining-----------')
model.fit(cnn_data_train,cnn_data_train_label,epochs=10,batch_size=32)
model.summary()
print('\nTesting------------')
loss,accuracy=model.evaluate(cnn_data_test,cnn_data_test_label)


print('test loss: ', loss)
print('test accuracy: ', accuracy)
t2 = time()
print("time used:",t2 - t1)

predict_03to05 = model.predict(data_sample_03to05)
data_new_outtxt = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/CNN_features/CNN_3years/CNN_3years_35_35/CNN_features_3years_35_35_03to05.txt'
np.savetxt(data_new_outtxt, predict_03to05, fmt='%s', newline='\n')