import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from osgeo import gdal
from sklearn import preprocessing

# Read land use classification layer data
file_land2000 = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/Landuse_final/lan2000_final.tif'
data_land2000 = gdal.Open(file_land2000)

im_height = data_land2000.RasterYSize  # Number of rows in the raster matrix
im_width = data_land2000.RasterXSize  # Number of columns in the raster matrix

# Land use data by year
im_data_land2000 = data_land2000.ReadAsArray(0, 0, im_width, im_height)

# Prepare initial cluster data
#
# # Input impact factor data: an array of Cluster_SH * 17, these 17 columns are spatial impact factors
Cluster_SH = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalData/AllFactors/AllFactors_20102017TXT.txt')

# Normalized function
min_max_scaler = preprocessing.MinMaxScaler()
Cluster_SH_Norm = min_max_scaler.fit_transform(Cluster_SH)
Cluster_SH_Norm_mat = np.mat(Cluster_SH_Norm)

print("shape of Cluster_SH_Norm_mat:",Cluster_SH_Norm_mat.shape)

# Construct NoBinary clusterer
estimator_NoBinary = KMeans(n_clusters=16)# Construct NoBinary clusterer
estimator_NoBinary.fit(Cluster_SH_Norm_mat)# clusterer
label_pred_NoBinary = estimator_NoBinary.labels_ # Get cluster labels
centroids_NoBinary = estimator_NoBinary.cluster_centers_ # Get Cluster Center
inertia_NoBinary = estimator_NoBinary.inertia_ # Get the sum of the clustering criteria

Cluster_result_NoBinary = np.zeros((im_height, im_width))
for row in range(0, im_height):
    for col in range(0, im_width):
        Cluster_result_NoBinary[row, col] = 100

index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2000[row][col] != 0:
            Cluster_result_NoBinary[row][col] = label_pred_NoBinary[index]
            index = index + 1

np.savetxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Clustering/K-Means_cluster.txt', Cluster_result_NoBinary, fmt='%s', newline='\n')