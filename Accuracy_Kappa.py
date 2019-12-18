import numpy as np
import random
from osgeo import gdal
import csv


def Confusion_Cal(im_data_refer, data_new):
    # Calculate the width and length of the image
    im_height = im_data_refer.shape[0]
    im_width = im_data_refer.shape[1]

    # Calculate the confusion matrix and Kappa coefficient
    Confusion_matrix = np.zeros((6, 6))
    for row in range(12, im_height - 12):
        for col in range(12, im_width - 12):
            if im_data_refer[row][col] != 0:
                if im_data_refer[row][col] == 1:
                    if data_new[row][col] == 1:
                        Confusion_matrix[0][0] = Confusion_matrix[0][0] + 1
                    elif data_new[row][col] == 2:
                        Confusion_matrix[0][1] = Confusion_matrix[0][1] + 1
                    elif data_new[row][col] == 3:
                        Confusion_matrix[0][2] = Confusion_matrix[0][2] + 1
                    elif data_new[row][col] == 4:
                        Confusion_matrix[0][3] = Confusion_matrix[0][3] + 1
                    elif data_new[row][col] == 5:
                        Confusion_matrix[0][4] = Confusion_matrix[0][4] + 1
                    elif data_new[row][col] == 6:
                        Confusion_matrix[0][5] = Confusion_matrix[0][5] + 1

                if im_data_refer[row][col] == 2:
                    if data_new[row][col] == 1:
                        Confusion_matrix[1][0] = Confusion_matrix[1][0] + 1
                    elif data_new[row][col] == 2:
                        Confusion_matrix[1][1] = Confusion_matrix[1][1] + 1
                    elif data_new[row][col] == 3:
                        Confusion_matrix[1][2] = Confusion_matrix[1][2] + 1
                    elif data_new[row][col] == 4:
                        Confusion_matrix[1][3] = Confusion_matrix[1][3] + 1
                    elif data_new[row][col] == 5:
                        Confusion_matrix[1][4] = Confusion_matrix[1][4] + 1
                    elif data_new[row][col] == 6:
                        Confusion_matrix[1][5] = Confusion_matrix[1][5] + 1

                if im_data_refer[row][col] == 3:
                    if data_new[row][col] == 1:
                        Confusion_matrix[2][0] = Confusion_matrix[2][0] + 1
                    elif data_new[row][col] == 2:
                        Confusion_matrix[2][1] = Confusion_matrix[2][1] + 1
                    elif data_new[row][col] == 3:
                        Confusion_matrix[2][2] = Confusion_matrix[2][2] + 1
                    elif data_new[row][col] == 4:
                        Confusion_matrix[2][3] = Confusion_matrix[2][3] + 1
                    elif data_new[row][col] == 5:
                        Confusion_matrix[2][4] = Confusion_matrix[2][4] + 1
                    elif data_new[row][col] == 6:
                        Confusion_matrix[2][5] = Confusion_matrix[2][5] + 1

                if im_data_refer[row][col] == 4:
                    if data_new[row][col] == 1:
                        Confusion_matrix[3][0] = Confusion_matrix[3][0] + 1
                    elif data_new[row][col] == 2:
                        Confusion_matrix[3][1] = Confusion_matrix[3][1] + 1
                    elif data_new[row][col] == 3:
                        Confusion_matrix[3][2] = Confusion_matrix[3][2] + 1
                    elif data_new[row][col] == 4:
                        Confusion_matrix[3][3] = Confusion_matrix[3][3] + 1
                    elif data_new[row][col] == 5:
                        Confusion_matrix[3][4] = Confusion_matrix[3][4] + 1
                    elif data_new[row][col] == 6:
                        Confusion_matrix[3][5] = Confusion_matrix[3][5] + 1

                if im_data_refer[row][col] == 5:
                    if data_new[row][col] == 1:
                        Confusion_matrix[4][0] = Confusion_matrix[4][0] + 1
                    elif data_new[row][col] == 2:
                        Confusion_matrix[4][1] = Confusion_matrix[4][1] + 1
                    elif data_new[row][col] == 3:
                        Confusion_matrix[4][2] = Confusion_matrix[4][2] + 1
                    elif data_new[row][col] == 4:
                        Confusion_matrix[4][3] = Confusion_matrix[4][3] + 1
                    elif data_new[row][col] == 5:
                        Confusion_matrix[4][4] = Confusion_matrix[4][4] + 1
                    elif data_new[row][col] == 6:
                        Confusion_matrix[4][5] = Confusion_matrix[4][5] + 1

                if im_data_refer[row][col] == 6:
                    if data_new[row][col] == 1:
                        Confusion_matrix[5][0] = Confusion_matrix[5][0] + 1
                    elif data_new[row][col] == 2:
                        Confusion_matrix[5][1] = Confusion_matrix[5][1] + 1
                    elif data_new[row][col] == 3:
                        Confusion_matrix[5][2] = Confusion_matrix[5][2] + 1
                    elif data_new[row][col] == 4:
                        Confusion_matrix[5][3] = Confusion_matrix[5][3] + 1
                    elif data_new[row][col] == 5:
                        Confusion_matrix[5][4] = Confusion_matrix[5][4] + 1
                    elif data_new[row][col] == 6:
                        Confusion_matrix[5][5] = Confusion_matrix[5][5] + 1

    return Confusion_matrix


def Multilabel_Precision_Recall(confusion_matrix):
    Precision_1 = confusion_matrix[0][0] / (
                confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0] +
                confusion_matrix[4][0] + confusion_matrix[5][0])
    Recall_1 = confusion_matrix[0][0] / (
                confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3] +
                confusion_matrix[0][4] + confusion_matrix[0][5])
    F1score_1 = 2 * Precision_1 * Recall_1 / (Precision_1 + Recall_1)

    Precision_2 = confusion_matrix[1][1] / (
                confusion_matrix[0][1] + confusion_matrix[1][1] + confusion_matrix[2][1] + confusion_matrix[3][1] +
                confusion_matrix[4][1] + confusion_matrix[5][1])
    Recall_2 = confusion_matrix[1][1] / (
                confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[1][3] +
                confusion_matrix[1][4] + confusion_matrix[1][5])
    F1score_2 = 2 * Precision_2 * Recall_2 / (Precision_2 + Recall_2)

    Precision_3 = confusion_matrix[2][2] / (
                confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[2][2] + confusion_matrix[3][2] +
                confusion_matrix[4][2] + confusion_matrix[5][2])
    Recall_3 = confusion_matrix[2][2] / (
                confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2] + confusion_matrix[2][3] +
                confusion_matrix[2][4] + confusion_matrix[2][5])
    F1score_3 = 2 * Precision_3 * Recall_3 / (Precision_3 + Recall_3)

    Precision_4 = confusion_matrix[3][3] / (
                confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3] + confusion_matrix[3][3] +
                confusion_matrix[4][3] + confusion_matrix[5][3])
    Recall_4 = confusion_matrix[3][3] / (
                confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2] + confusion_matrix[3][3] +
                confusion_matrix[3][4] + confusion_matrix[3][5])
    F1score_4 = 2 * Precision_4 * Recall_4 / (Precision_4 + Recall_4)

    Precision_5 = confusion_matrix[4][4] / (
                confusion_matrix[0][4] + confusion_matrix[1][4] + confusion_matrix[2][4] + confusion_matrix[3][4] +
                confusion_matrix[4][4] + confusion_matrix[5][4])
    Recall_5 = confusion_matrix[4][4] / (
                confusion_matrix[4][0] + confusion_matrix[4][1] + confusion_matrix[4][2] + confusion_matrix[4][3] +
                confusion_matrix[4][4] + confusion_matrix[4][5])
    F1score_5 = 2 * Precision_5 * Recall_5 / (Precision_5 + Recall_5)

    Precision_6 = confusion_matrix[5][5] / (
                confusion_matrix[0][5] + confusion_matrix[1][5] + confusion_matrix[2][5] + confusion_matrix[3][5] +
                confusion_matrix[4][5] + confusion_matrix[5][5])
    Recall_6 = confusion_matrix[5][5] / (
                confusion_matrix[5][0] + confusion_matrix[5][1] + confusion_matrix[5][2] + confusion_matrix[5][3] +
                confusion_matrix[5][4] + confusion_matrix[5][5])
    F1score_6 = 2 * Precision_6 * Recall_6 / (Precision_6 + Recall_6)

    PRF = np.concatenate(([Precision_1, Recall_1, F1score_1],
                          [Precision_2, Recall_2, F1score_2],
                          [Precision_3, Recall_3, F1score_3],
                          [Precision_4, Recall_4, F1score_4],
                          [Precision_5, Recall_5, F1score_5],
                          [Precision_6, Recall_6, F1score_6]), axis=0).reshape(6, 3)

    Accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3] +
                confusion_matrix[4][4] + confusion_matrix[5][5]) / np.sum(confusion_matrix)
    Pe = (np.sum(confusion_matrix[0, :]) * np.sum(confusion_matrix[:, 0]) + np.sum(confusion_matrix[1, :]) * np.sum(
        confusion_matrix[:, 1]) + np.sum(confusion_matrix[2, :]) * np.sum(confusion_matrix[:, 2])
          + np.sum(confusion_matrix[3, :]) * np.sum(confusion_matrix[:, 3]) + np.sum(confusion_matrix[4, :]) * np.sum(
                confusion_matrix[:, 4]) + np.sum(confusion_matrix[5, :]) * np.sum(confusion_matrix[:, 5])) / (
                     np.sum(confusion_matrix) * np.sum(confusion_matrix))

    Kappa = (Accuracy - Pe) / (1 - Pe)

    macro_score = (F1score_1 + F1score_2 + F1score_3 + F1score_4 + F1score_5 + F1score_6) / 6

    return PRF, Accuracy, Kappa, macro_score

# Actual LU map
file_land_2015 = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/lan2015_final.tif'
data_land_2015 = gdal.Open(file_land_2015)

im_height = data_land_2015.RasterYSize  # Number of rows in the raster matrix
im_width = data_land_2015.RasterXSize  # Number of columns in the raster matrix

im_data_land_2015 = data_land_2015.ReadAsArray(0, 0, im_width, im_height)

# simulation results of ANN_CA
data_ANN_whole = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Conventional_Models/ANN_whole.txt')

# Calculate the confusion matrix for ANN model 2015 WholeArea
confusion_matrix_ANN_whole = Confusion_Cal(im_data_land_2015,data_ANN_whole)

PRF_ANN_whole,Accuracy_ANN_whole,Kappa_ANN_whole,macro_score_ANN_whole = Multilabel_Precision_Recall(confusion_matrix_ANN_whole)
print("PRF_ANN_whole:",PRF_ANN_whole)
print("Accuracy_ANN_whole:",Accuracy_ANN_whole)
print("Kappa_ANN_whole:",Kappa_ANN_whole)
print("macro_score_ANN_whole:",macro_score_ANN_whole)

path = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/accuracy_data_new.csv'
f = open(path,'a+',encoding='utf-8')
csv_write = csv.writer(f)
csv_write.writerow(['ANN_whole',Accuracy_ANN_whole,Kappa_ANN_whole,macro_score_ANN_whole])
f.close()