# Land-use-change-simulation-with-PST-CA
A novel hybrid cellular automata model coupling area partitioning and spatiotemporal neighborhood features for LUC simulation Requirements (1) Python 3 (2) Tensorflow > 1.2 (3) Keras > 2.1.1 (4) Numpy

Data (1) Time-series land use maps (2) Factors that driving land use changing such as accessibility to transport lines, accessibility to public infrastructures, and terrain conditions, which may include distance to motorway, trunk, primary road, minor road, railway, river; distance to city center, county center, town center, airport, train station, railway station, coach station, bus station, ferry station; Elevation and slope.

Highlights (1) Self-organizing map was two different models to divide entire study area into sseveral relatively homogeneous sub-regions, which may address spatial heterogeneity existing in LUC. (2) 3D CNN was employed to capture spatiotemporal features contained among time-series land use maps.

Models (1) K-means_cluster.py and SOM_cluster were two different methods used to partition entire study area. (2) LR_CA, SVM_CA, RF_CA and ANN_CA were four traditional models used for LUC simulation. (3) PST_CA was our proposed model, which coupled area partitioning and spatiotemporal neighborhood features for LUC simulation. (4) Accuracy_Kappa.py and FoM.py were used to assess the performance of these models.
