import numpy as np
import pylab as pl
from osgeo import gdal
from sklearn import preprocessing

class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X:  The shape is N * D， there are N input samples, each D dimension
        :param output: (n, m) a tuple whose shape of the output layer is a two-dimensional matrix of n * m
        :param iteration: Number of iterations
        :param batch_size:Number of samples per iteration
        Initialize a weight matrix with the shape D * (n * m), that is, there are n * m weight vectors, each D dimension
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])
        print(self.W.shape)

    def GetN(self, t):
        """
        :param t: Time t, here the number of iterations is used to represent time
        :return: Returns an integer representing the topological distance. The larger the time, the smaller the topological neighborhood.
        """
        a = min(self.output)
        return int(a - float(a) * t / self.iteration)

    def Geteta(self, t, n):
        """
        :param t: Time t, here the number of iterations is used to represent time
        :param n: topological distance
        :return: return learning rate
        """
        return np.power(np.e, -n) / (t + 2)

    def updata_W(self, X, t, winner):
        N = self.GetN(t)
        for x, i in enumerate(winner):
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))

    def getneighbor(self, index, N):
        """
        :param index:Index of the winning neuron
        :param N: Neighborhood radius
        :return ans: Returns a list of sets of neuron coordinates that need to be updated in different neighborhood radii
        """
        a, b = self.output
        length = a * b

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)

        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N: ans[max(dist_a, dist_b)].add(i)
        return ans

    def train(self):
        """
        train_Y: The training samples and shape are batch_size * (n * m)
        winner: A one-dimensional vector, the index of the batch_size winning neurons
        :return: The return value is adjusted W
        """
        count = 0
        while self.iteration > count:
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            normal_W(self.W)
            normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()
            self.updata_W(train_X, count, winner)
            count += 1
        return self.W

    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        # print(winner)
        return winner

def normal_X(X):
    """
    :param X: 2D matrix, N * D, N D-dimensional data
    :return: Normalize X results
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X


def normal_W(W):
    """
    :param W: 2D matrix, D*(n*m)
    :return: Normalize
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W


# Read land use classification layer data
file_land2000 = '/home/anaconda_workspace/qyh/SH_100m/Landuse_final/lan2000_final.tif'
data_land2000 = gdal.Open(file_land2000)

im_height = data_land2000.RasterYSize  # Number of rows in the raster matrix
im_width = data_land2000.RasterXSize  # Number of columns in the raster matrix

# Land use data by year
im_data_land2000 = data_land2000.ReadAsArray(0, 0, im_width, im_height)

# Prepare initial cluster data

# Input impact factor data: an array of Cluster_SH * 17, these 17 columns are spatial impact factors
Cluster_SH = np.loadtxt('/home/anaconda_workspace/qyh/SH_100m/AllFactors/AllFactors_20102017TXT.txt')

# Normalized function
min_max_scaler = preprocessing.MinMaxScaler()
Cluster_SH_Norm = min_max_scaler.fit_transform(Cluster_SH)
print("shape of Cluster_SH_Norm:",Cluster_SH_Norm.shape)

Cluster_SH_Norm_NoBinary = np.mat(Cluster_SH_Norm)
som_NoBinary = SOM(Cluster_SH_Norm_NoBinary, (2, 3), 16, 64)
som_NoBinary.train()
res_NoBinary = som_NoBinary.train_result()

classify_result_NoBinary = {}
# classify_result is a dictionary that stores keywords and corresponding categories under the keyword,
# win [0] represents the category label, and i represents the number
for i, win in enumerate(res_NoBinary):
    if not classify_result_NoBinary.get(win[0]):
        classify_result_NoBinary.setdefault(win[0], [i])
    else:
        classify_result_NoBinary[win[0]].append(i)

number = len(res_NoBinary)

Label_NoBinary = np.empty((number,1))
for i in range(number):
    # temp is the label after clustering
    temp = res_NoBinary[i][0]
    Label_NoBinary[i,0] = int(temp)

# Get clustered array data
data_new_NoBinary = np.empty((im_height, im_width))

# To avoid confusion with clustering "0", initialize to 100
for row in range(im_height):
    for col in range(im_width):
        data_new_NoBinary[row, col] = 6

index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2000[row][col] != 0:
            data_new_NoBinary[row][col] = Label_NoBinary[index][0]
            index = index + 1

np.savetxt('/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/Partition_Merely/SOM_cluster.txt', data_new_NoBinary, fmt='%s', newline='\n')