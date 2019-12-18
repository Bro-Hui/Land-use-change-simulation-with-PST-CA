import numpy as np
import time
import csv
import random
from osgeo import gdal
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Reshape
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, PReLU,  LSTM, multiply, concatenate

# Read land use classification layer data
file_land2005 = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/lan2005_final.tif'
data_land2005 = gdal.Open(file_land2005)

file_land2010 = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/lan2010_final.tif'
data_land2010 = gdal.Open(file_land2010)

file_land2015 = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/lan2015_final.tif'
data_land2015 = gdal.Open(file_land2015)

im_height = data_land2005.RasterYSize  # Number of rows in the raster matrix
im_width = data_land2005.RasterXSize  # Number of columns in the raster matrix

# Land use data
im_data_land2005 = data_land2005.ReadAsArray(0, 0, im_width, im_height)
im_data_land2010 = data_land2010.ReadAsArray(0, 0, im_width, im_height)
im_data_land2015 = data_land2015.ReadAsArray(0, 0, im_width, im_height)

file_partition = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/im_SOM_final.tif'
data_partition = gdal.Open(file_partition)

partition_result = data_partition.ReadAsArray(0, 0, im_width, im_height)

Category_0 = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        # partition_result “100” denotes the pixel is outside the study area
        if partition_result[row][col] != 100:
            if partition_result[row][col] == 0:
                Category_0 += 1

print("Category_0:", Category_0)

part_zero_samples_2005 = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Samples/Partition_Merely/2005/part_zero_samples_2005.txt')
part_zero_labels_2005 = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Samples/Partition_Merely/2005/part_zero_labels_2005.txt')

part_zero_CNN1year_2005  = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/CNN_features/CNN_1year/CNN_1year_11_11/part_zero_CNN_features_1year_11_11_2005.txt')

part_zero_Sample_2005_UnNorm = np.concatenate((part_zero_samples_2005,part_zero_CNN1year_2005),axis=1)
# 归一化函数
min_max_scaler = preprocessing.MinMaxScaler()
part_zero_Sample_2005_final = min_max_scaler.fit_transform(part_zero_Sample_2005_UnNorm)

print("the shape of part_zero_Sample_2005_final:",part_zero_Sample_2005_final.shape)
print("the shape of part_zero_labels_2005:",part_zero_labels_2005.shape)

part_zero_samples_2010 = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Samples/Partition_Merely/2010/part_zero_samples_2010.txt')
part_zero_labels_2010 = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Samples/Partition_Merely/2010/part_zero_labels_2010.txt')

part_zero_CNN1year_2010  = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/CNN_features/CNN_1year/CNN_1year_11_11/part_zero_CNN_features_1year_11_11_2010.txt')

part_zero_Sample_2010_UnNorm = np.concatenate((part_zero_samples_2010,part_zero_CNN1year_2010),axis=1)
# Normalization function
min_max_scaler = preprocessing.MinMaxScaler()
part_zero_Sample_2010_final = min_max_scaler.fit_transform(part_zero_Sample_2010_UnNorm)

print("the shape of part_zero_Sample_2010_final:",part_zero_Sample_2010_final.shape)
print("the shape of part_zero_labels_2010:",part_zero_labels_2010.shape)

# acquire training samples
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []

    for i in range(length):
        random_list.append(random.randint(start, stop))
    return np.array(random_list)


random_list = random_int_list(0,part_zero_Sample_2005_final.shape[0]-1,int(part_zero_Sample_2005_final.shape[0]* 0.2))
ann_data = np.zeros((int(part_zero_Sample_2005_final.shape[0]* 0.2),30))
ann_data_label = np.zeros((int(part_zero_Sample_2005_final.shape[0]* 0.2),1))
for i in range(int(part_zero_Sample_2005_final.shape[0]* 0.2)):
    temp = random_list[i]
    ann_data[i]= part_zero_Sample_2005_final[temp]               #ann_data: 20% of origin samples
    ann_data_label[i] = part_zero_labels_2005[temp]

train_num = int(ann_data.shape[0] * 0.7)    # set number of training samples
test_num = ann_data.shape[0] - train_num    # set number of test samples

# acquire training samples from ann_data
ann_data_train = np.zeros((train_num,30))
ann_data_test = np.zeros((test_num,30))
ann_data_train_label = np.zeros((train_num,1))
ann_data_test_label = np.zeros((test_num,1))
for i in range(train_num):
    ann_data_train[i] = ann_data[i]
    ann_data_train_label[i] = ann_data_label[i]
for j in range(train_num,ann_data.shape[0]):
    ann_data_test[j - train_num] = ann_data[j]
    ann_data_test_label[j - train_num] = ann_data_label[j]

# transform label into one-hot
ann_data_train_label = np_utils.to_categorical(ann_data_train_label, num_classes=7)
ann_data_test_label = np_utils.to_categorical(ann_data_test_label, num_classes=7)

# set neural networks
model = Sequential([
    Dense(32, input_dim=30),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dropout(0.02),
    Dense(7),
    Activation('softmax'),
])

# select optimizer
rmsprop = RMSprop(lr=0.0001, rho=0.8, epsilon=1e-08, decay=0.0)

# The loss function and the precision evaluation index are selected
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Start training using training data
print('Training ------------')

model.fit(ann_data_train, ann_data_train_label, epochs=10, batch_size=64)

# Optimize with test data
print('\nTesting ------------')

loss, accuracy = model.evaluate(ann_data_test, ann_data_test_label)
# Output training results
print('test loss: ', loss)
print('test accuracy: ', accuracy)

# The simulation was performed using 2010 image data
predict = model.predict(part_zero_Sample_2010_final, batch_size=32, verbose=0)

predict_out = np.zeros((predict.shape[0], predict.shape[1]))
# Find the maximum value for each row and determine if it is greater than the threshold
theshold = 0.8
Change_sum = 0
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        if predict[i][j] >= theshold:
            predict_out[i][j] = 1
            Change_sum = Change_sum + 1
        else:
            predict_out[i][j] = 0

# for i in range(predict.shape[0]):
#     for j in range(predict.shape[1]):
#         if j == 4:
#             predict_out[i][j] = 1
#             Change_sum = Change_sum + 1
#         else:
#             predict_out[i][j] = 0

print("The number of predict > theshold:", Change_sum)
print("The rate of predict > theshold:", Change_sum / Category_0)

Cos = np.zeros((6, 7))
Cos = [[0, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 1, 0, 1],
       [0, 0, 0, 0, 0, 0, 1],
       [0, 1, 1, 1, 1, 1, 1]]

Change_sum_new = 0
for i in range(predict_out.shape[0]):
    # Using the initial year of the simulation as a starting point, determine whether transformation can occur
    temp_label = int(part_zero_labels_2005[i])
    predict_out[i] = np.multiply(predict_out[i], Cos[temp_label - 1])
    Change_sum_new = Change_sum_new + np.sum(predict_out[i])

print("The number of predict > theshold and changeable:", Change_sum_new)
print("The rate of predict > theshold and changeable:", Change_sum_new / Category_0)

# acquire new labels
Label_predict = np.zeros((Category_0, 1))

index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if partition_result[row][col] != 100:
            if partition_result[row][col] == 0:
                Label_predict[index][0] = im_data_land2010[row][col]

                for j in range(predict.shape[1]):
                    if predict_out[index][j] == 1:
                        Label_predict[index][0] = j

                index += 1

assert (index == Category_0)

same_label_origin = 0
for i in range(Category_0):
    if part_zero_labels_2005[i] == part_zero_labels_2010[i]:
        same_label_origin += 1

print("The same label of partition one between im_data_land2010 and im_data_land2015 = ", same_label_origin)

same_label = 0
for i in range(Category_0):
    if part_zero_labels_2005[i] == Label_predict[i,0]:
        same_label += 1

print("The same label of partition one between im_data_land2010and Label_predict = ", same_label)

same = 0
for i in range(Category_0):
    if part_zero_labels_2010[i] == Label_predict[i,0]:
        same += 1

print("The same label of partition one between im_data_land2015 and Label_predict = ", same)
print("the accuracy of prediction is:",same/Category_0)

np.savetxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Partition_CNNs/part_zero/Label_predict_part_zero_CNN1year.txt', Label_predict, fmt='%s', newline='\n')