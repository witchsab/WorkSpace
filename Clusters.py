# ----------------------- Clustering  SIFT  ---------------------
import numpy as np 
from sklearn.cluster import MiniBatchKMeans


# RESHAPE To np array :   dataframe to FV 
XD = list(mydataSIFT['siftdes'])

# 3. Works as # 1, 2
XA = np.asarray(XD)
X=np.concatenate(XA, axis=0)


# # 1. Works as # 2.
# dlist = [] 
# for item in XD: 
#     dlist.extend(item)
# LDP = np.array(dlist)

# # 2. Works as # 1
# descriptors = np.array([])
# for des in XD:
#     descriptors = np.append(descriptors, np.array(des))
# desc = np.reshape(descriptors, (int(len(descriptors)/128), 128))
# desc2 = np.float32(desc)


# Cluster the descriptors 
n_clusters = 200
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(X)
# y_means = kmeans.predict (X)


n_bins = 2000
# Create FV Histogram from BOvW
feature_vectors=[]
class_vectors=[]
for item in XD: 
    predict_kmeans=kmeans.predict(item)
    #calculates the histogram
    hist, bin_edges=np.histogram(predict_kmeans, bins=n_bins)
    #histogram is the feature vector
    feature_vectors.append(hist)


# from sklearn.neighbors import NearestNeighbors
# neighbor = NearestNeighbors(n_neighbors = 30)
# neighbor.fit(feature_vectors)


from sklearn.neighbors import KDTree
SIFTtree = KDTree(feature_vectors)


# ----------- search 

q_path = random.sample(imagepaths, 1)[0]
print (q_path)

import ImageSearch_Algo_SIFT
# get the feature  
q_kp, q_des = ImageSearch_Algo_SIFT.FEATURE (q_path)

predict_kmeans=kmeans.predict(q_des)
#calculates the histogram
hist1, bin_edges1=np.histogram(predict_kmeans, bins=n_bins)
#histogram is the feature vector
q_feature_vector = hist1


# ------- Using KD TREE
# reshape 
F = q_feature_vector.reshape (1, -1)
dist, result = SIFTtree.query(F, k=50)
print (result)
flist = list (mydataSIFT.iloc[ result[0].tolist()]['file'])
slist = list (dist[0])
matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 

a, q, pos, cnt = accuracy.accuracy_matches(q_path, matches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)



# # using nearest neighbor
# dist, result = neighbor.kneighbors([q_feature_vector])
# print (result)

# flist = list (mydataSIFT.iloc[ result[0].tolist()]['file'])
# slist = list (dist[0])
# matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 

# a, q, pos, cnt = accuracy.accuracy_matches(q_path, matches, 20)
# print('Accuracy =',  a, '%', '| Quality:', q)
# print('Count', cnt, ' | position', pos)


















# ----------------------- Clustering  RGB ---------------------

from sklearn.neighbors import KDTree
import time 

# RESHAPE To np array :   dataframe to FV 
XD = list(mydataRGB['imagehist'])
XA = np.asarray(XD)
nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

# Tree Fit 
RGBtree = KDTree( X, metric='manhattan')


# search : 
ft = ImageSearch_Algo_RGB.RGB_FEATURE (q_path) 
nx, ny, nz = ft.shape  # know the shape before you flatten
F = ft.reshape ((1, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 
scores, ind = RGBtree.query(F, k=50)


# --------------------  Clustering  HSV ---------------------

from sklearn.neighbors import KDTree
import numpy as np
import time 

# RESHAPE To np array :   dataframe to FV 
YD = list(mydataHSV['imagehist'])
X = np.asarray(YD)
# nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
# X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

# Tree Fit 
HSVtree = KDTree( X ) # , metric='euclidean')

# search : 
fh = ImageSearch_Algo_HSV.HSV_FEATURE (q_path)
fh = np.asarray(fh)
F = fh.reshape (1, -1) # gives a 2 D matice (sample, value) which can be fed to KMeans 

scores, ind = HSVtree.query(F, k=100)




# --------------------  Clustering  HASH ---------------------

# RESHAPE To np array :   dataframe to FV 
YD = list(mydataHASH['ahash'])
result_array = []
for item in YD : 
    onearray = np.asarray(np.array (item.hash), dtype=float)
    result_array.append(onearray)
YA = np.asarray(result_array)
nsamples, x, y = YA.shape  # know the shape before you flatten
X = YA.reshape ( nsamples, x*y ) # gives a 2 D matice (sample, value) which can be fed to KMeans 

# Tree Fit 
HASHTree = KDTree( X ,  metric='euclidean')

# search : 
fh = np.array (ImageSearch_Algo_Hash.HASH_FEATURE(q_path, 'ahash', 16).hash)
fd = np.asarray(fh , dtype=float) # since hash is an array of bool -> numpy its 1,0
x, y = fd.shape # know the shape before you flatten
FF = fd.reshape (1, x*y) # gives a 2 D matice (sample, value) which can be fed to KMeans 
scores, ind = HASHTree.query(FF, k=100)
