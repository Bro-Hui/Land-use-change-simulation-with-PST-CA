import numpy as np
import time
import csv
import random
from osgeo import gdal
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils

# Read land use classification layer data
file_land2005 = '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Landuse_final/lan2005_final.tif'
data_land2005 = gdal.Open(file_land2005)

file_land2010 = '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Landuse_final/lan2010_final.tif'
data_land2010 = gdal.Open(file_land2010)

file_land2015 = '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Landuse_final/lan2015_final.tif'
data_land2015 = gdal.Open(file_land2015)

im_height = data_land2005.RasterYSize  # Number of rows in the raster matrix
im_width = data_land2005.RasterXSize  # Number of columns in the raster matrix

# Land use data
im_data_land2005 = data_land2005.ReadAsArray(0, 0, im_width, im_height)
im_data_land2010 = data_land2010.ReadAsArray(0, 0, im_width, im_height)
im_data_land2015 = data_land2015.ReadAsArray(0, 0, im_width, im_height)

number = 0
# Number of pixels in Shanghai
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
            if im_data_land2005[row][col] != 0 :
                number = number + 1

print("number of ShangHai:\n",number)

# train samples are composed of AllFactors、Neighborhood features in 2005
All_Factors_2005 = np.loadtxt(
    '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/AllFactors/AllFactors_20102017TXT.txt')

Neighbor_11_11_2005 = np.loadtxt(
    '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Neighbor_field/Neighbor_field11_11/im_field_2005.txt')

# samples and labels in 2005
Sample_2005 = np.concatenate((All_Factors_2005, Neighbor_11_11_2005), axis=1)

# Normalization function
min_max_scaler = preprocessing.MinMaxScaler()
Sample_2005_Norm = min_max_scaler.fit_transform(Sample_2005)

Label_2005 = np.empty((number, 1))
index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            Label_2005[index, 0] = im_data_land2010[row][col]
            index += 1

print("the shape of Sample_2005_Norm:", Sample_2005_Norm.shape)
print("the shape of Label_2005:", Label_2005.shape)

# train samples are composed of AllFactors、Neighborhood features in 2010
All_Factors_2010 = np.loadtxt(
    '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/AllFactors/AllFactors_20102017TXT.txt')

Neighbor_11_11_2010 = np.loadtxt(
    '/home/kwan/Workspace/qyh/data/SH_100m/ExperimentalData/Neighbor_field/Neighbor_field11_11/im_field_2010.txt')

# samples and labels in 2010
Sample_2010 = np.concatenate((All_Factors_2010, Neighbor_11_11_2010), axis=1)

# Normalization function
min_max_scaler = preprocessing.MinMaxScaler()
Sample_2010_Norm = min_max_scaler.fit_transform(Sample_2010)

Label_2010 = np.empty((number, 1))
index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2015[row][col] != 0:
            Label_2010[index, 0] = im_data_land2015[row][col]
            index += 1

print("the shape of Sample_2010_Norm:", Sample_2010_Norm.shape)
print("the shape of Label_2010:", Label_2010.shape)

# acquire training samples
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []

    for i in range(length):
        random_list.append(random.randint(start, stop))
    return np.array(random_list)


random_list = random_int_list(0,Sample_2005_Norm.shape[0]-1,int(Sample_2005_Norm.shape[0]* 0.2))

ann_data = np.zeros((int(Sample_2005_Norm.shape[0]* 0.2),23))
ann_data_label = np.zeros((int(Sample_2005_Norm.shape[0]* 0.2),1))
for i in range(int(Sample_2005_Norm.shape[0]* 0.2)):
    temp = random_list[i]
    ann_data[i]= Sample_2005_Norm[temp]               #ann_data: 20% of origin samples
    ann_data_label[i] = Label_2005[temp]         #ann_data_label

train_num = int(ann_data.shape[0] * 0.7)    # set number of training samples
test_num = ann_data.shape[0] - train_num    # set number of test samples

# acquire training samples from ann_data
ann_data_train = np.zeros((train_num,23))
ann_data_test = np.zeros((test_num,23))
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

lr = LogisticRegression()
lr.fit(ann_data_train, ann_data_train_label)

Label_predict = lr.predict(Sample_2010_Norm)

# Gets the simulated array data
data_new = np.zeros((im_height, im_width))
index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            data_new[row][col] = Label_predict[index][0]
            index = index + 1

same_label_origin = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            if im_data_land2010[row][col] == im_data_land2015[row][col]:
                same_label_origin = same_label_origin + 1

print("The same label between im_data_land2010 and im_data_land2015 = ", same_label_origin)

same_label = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            if im_data_land2010[row][col] == data_new[row][col]:
                same_label = same_label + 1

print("The same label between im_data_land2010 and data_new = ", same_label)

same = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            if im_data_land2015[row][col] == data_new[row][col]:
                same = same + 1

print("The same label between im_data_land2015 and data_new = ", same)
print("the accuracy of predict is:", same / number)

np.savetxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Conventional_Models/Logistic_Regression_CA.txt', data_new, fmt='%s', newline='\n')