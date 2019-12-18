import numpy as np
import random
from osgeo import gdal

# 真实结果
file_land_2015 = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/lan2015_final.tif'
data_land_2015 = gdal.Open(file_land_2015)

im_height = data_land_2015.RasterYSize  # Number of rows in the raster matrix
im_width = data_land_2015.RasterXSize  # Number of columns in the raster matrix

im_data_land_2015 = data_land_2015.ReadAsArray(0, 0, im_width, im_height)

# simulation results of conventional models in 2015
file_LR = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Conventional_Models/Logistic_Regression_CA.txt'
im_data_LR = np.loadtxt(file_LR)

file_SVM = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Conventional_Models/SVM_CA.txt'
im_data_SVM = np.loadtxt(file_SVM)

file_RF = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Conventional_Models/RF_CA.txt'
im_data_RF = np.loadtxt(file_RF)

file_ANN = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Conventional_Models/ANN_whole.txt'
im_data_ANN = np.loadtxt(file_ANN)


# simulation results of partition in 2015
file_K_Means = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Partition_Merely/K_means/im_K_means.txt'
im_data_K_Means = np.loadtxt(file_K_Means)

file_SOM = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Partition_Merely/im_SOM.txt'
im_data_SOM = np.loadtxt(file_SOM)


# simulation results of CNN in 2015
file_CNN1year = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/CNNs_Merely/ANN_CNN1year.txt'
im_data_CNN1year = np.loadtxt(file_CNN1year)

file_CNN2years = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/CNNs_Merely/ANN_CNN2years.txt'
im_data_CNN2years = np.loadtxt(file_CNN2years)

file_CNN3years = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/CNNs_Merely/ANN_CNN3years.txt'
im_data_CNN3years = np.loadtxt(file_CNN3years)

file_CNN4years = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/CNNs_Merely/ANN_CNN4years.txt'
im_data_CNN4years = np.loadtxt(file_CNN4years)

file_CNN5years = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/CNNs_Merely/ANN_CNN5years.txt'
im_data_CNN5years = np.loadtxt(file_CNN5years)


# simulation results of PST_CA in 2015
file_PST = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Partition_CNNs/im_PST.txt'
im_data_PST = np.loadtxt(file_PST)

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
            if im_data_land2015[row][col] != 0 :
                number = number + 1

print("number of Shanghai:\n",number)

error_1_PST_2015 = 0
error_2_PST_2015 = 0
error_3_PST_2015 = 0
correct_4_PST_2015 = 0
error_PST_2015 = 0

data_error_PST_2015 = np.zeros((im_height, im_width))
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            data_error_PST_2015[row][col] = 1

            # stay unchanged but simulate changed
            if im_data_land2015[row][col] == im_data_land2010[row][col] and im_data_PST[row][col] != \
                    im_data_land2010[row][col]:
                data_error_PST_2015[row][col] = 50
                error_1_PST_2015 = error_1_PST_2015 + 1

            # actual changed but simulate unchanged
            if im_data_land2015[row][col] != im_data_land2010[row][col] and im_data_PST[row][col] == \
                    im_data_land2010[row][col]:
                data_error_PST_2015[row][col] = 100
                error_2_PST_2015 = error_2_PST_2015 + 1

            # actual changed and simulate changed, but simulate wrongly
            if im_data_land2015[row][col] != im_data_land2010[row][col] and im_data_PST[row][col] != \
                    im_data_land2010[row][col] and im_data_PST[row][col] != im_data_land2015[row][col]:
                data_error_PST_2015[row][col] = 150
                error_3_PST_2015 = error_3_PST_2015 + 1

            # actual changed and simulate changed, and simulate correctly
            if im_data_land2015[row][col] != im_data_land2010[row][col] and im_data_PST[row][col] != \
                    im_data_land2010[row][col] and im_data_PST[row][col] == im_data_land2015[row][col]:
                data_error_PST_2015[row][col] = 200
                correct_4_PST_2015 = correct_4_PST_2015 + 1

            # The predicted results are inconsistent with the actual results, total errors
            if im_data_land2015[row][col] != im_data_PST[row][col]:
                error_PST_2015 = error_PST_2015 + 1

FOM = correct_4_PST_2015 / (error_1_PST_2015 + error_2_PST_2015 + error_3_PST_2015 + correct_4_PST_2015)
# FOM_2 = (error_3_PST_2015+correct_4_PST_2015)/(error_1_PST_2015+error_2_PST_2015+error_3_PST_2015+correct_4_PST_2015)

print("number of error_1_PST_2015:", error_1_PST_2015)
print("number of error_2_PST_2015:", error_2_PST_2015)
print("number of error_3_PST_2015:", error_3_PST_2015)
print("number of correct_4_PST_2015:", correct_4_PST_2015)
print("number of error_PST_2015:", error_PST_2015)

print("------------------------")
print("correct FOM:", FOM)
# print("change FOM_2:", FOM_2)

data_error_PST_2015_outtxt = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Errors analysis/data_error_PST_2015.txt'
np.savetxt(data_error_PST_2015_outtxt, data_error_PST_2015, fmt='%s', newline='\n')