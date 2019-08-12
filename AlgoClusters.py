# ----------------------- Clustering  SIFT  ---------------------
import numpy as np 
from sklearn.cluster import MiniBatchKMeans
import time 

# -----------------------Method #1 --------------------
# Some problem in the implementation (check)
# 

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
n_bins = 200
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(X)
# y_means = kmeans.predict (X)



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
# reshape - something wrong in this implementation 
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




# -----------------------Method #2 --------------------
# https://github.com/mayuri0192/Image-classification/blob/master/descriptors.py

import ImageSearch_Algo_SIFT
import numpy as np 
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
import time

### Train KMeans and define Feature Vectors 
# define cluster size 
n_clusters = 5000
# Concatenate all descriptors in the training set together
training_descs = list(mydataSIFT['siftdes'])
all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
all_train_descriptors = np.array(all_train_descriptors)

# define the cluster model 
cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
# train kmeans or other cluster model on those descriptors selected above
cluster_model.fit(all_train_descriptors)
print('done clustering. Using clustering model to generate BoW histograms for each image.')
# compute set of cluster-reduced words for each image
img_clustered_words = [cluster_model.predict(raw_words) for raw_words in training_descs]
# finally make a histogram of clustered word counts for each image. These are the final features.
img_bow_hist = np.array([np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])
# create Tree for histograms 
SIFTtree2 = KDTree(img_bow_hist)


# # to reload module: uncomment use the following
# %load_ext autoreload
# %autoreload 2


### Search an image in the set 
# sample a image
q_path = random.sample(imagepaths, 1)[0]
print (q_path)
# log time 
start = time.time()
# get the feature for this image 
q_kp, q_des = ImageSearch_Algo_SIFT.FEATURE (q_path)
# get bow cluster
q_clustered_words = cluster_model.predict(q_des) 
# get FV histogram  
q_bow_hist = np.array([np.bincount(q_clustered_words, minlength=n_clusters)])
# search the KDTree for nearest match
dist, result = SIFTtree2.query(q_bow_hist, k=100)
t= time.time() - start
print (result)
print ('SIFT Search Tree: ', t , ' secs')
flist = list (mydataSIFT.iloc[ result[0].tolist()]['file'])
slist = list (dist[0])
matches = tuple(zip( slist, flist)) # create a list of tuples frm 2 lists
a, q, pos, cnt = accuracy.accuracy_matches(q_path, matches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)



from sklearn.cluster import KMeans
# --------- find unsupervised cluster ID with KMEANS  
X = img_bow_hist
km = KMeans(n_clusters=200)
km.fit(X)
km.predict(X)
labels = km.labels_
print (labels)
# # update labels to original dataframe
mydataSIFT['clusterID'] = pd.DataFrame(labels)





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
print (ind)

# Cluster ID generations w/ KMEANS
YD = list(mydataHSV['imagehist'])
X = np.asarray(YD)
HSVCluster = KMeans(n_clusters=33)
HSVCluster.fit(X)
# predict and generate cluster labels/IDs
HSVCluster.predict(X)
labels = HSVCluster.labels_
# print (labels)

# update cluster labels/IDs to original dataframe
mydataHSV['clusterID'] = pd.DataFrame(labels)

# show Cluster IDs
mydataHSV.sort_values('file')[['file', 'clusterID']]


# Cluster generation with DB SCAN 
from sklearn.cluster import DBSCAN 
db = DBSCAN(eps=0.3, min_samples=10).fit(X) 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_ 
# Number of clusters in labels, ignoring noise if present. 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
mydataHSV['DBID'] = pd.DataFrame(labels)
mydataHSV.sort_values('file')[['file', 'clusterID', 'DBID']]


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# cluster the data into five clusters
dbscan = DBSCAN(eps=40, min_samples = 2)
dbscan.fit_predict(X_scaled)







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
