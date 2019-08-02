# Unsupervised Learning with KMeans Clusters

# https://towardsdatascience.com/clustering-based-unsupervised-learning-8d705298ae51
# https://medium.com/datadriveninvestor/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# # for feature vector matrices from features daataframe 
# XD = list(mydataRGB['imagehist'])
# XA = np.asarray(XD)
# nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
# X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 
# X = XD

#---------------------------- Determine cluster size----------------------------------

# find the appropriate cluster number
plt.figure(figsize=(10, 8))
from sklearn.cluster import KMeans
from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
X = iris.data

# Calculate clusters using Elbow criteria 

wcss = []
for i in range(1, 51):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 51), wcss)
# plt.plot(range(1, 11), elbowIndex)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# find elbow using Kneelocator package 
# https://github.com/arvkevi/kneed#find-knee

from kneed import KneeLocator
elbow = KneeLocator( list(range(1,51)), wcss, S=1.0, curve='convex', direction='decreasing')
print ('Detected Elbow cluster value :', elbow.knee)



#------------------------------------ K-Means (Simplest model)------------------------------------
# Additional READs
# https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-clusters-in-a-k-means-unsupervised-learning-model/


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# %matplotlib inline

from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
X = iris.data
#KMeans
km = KMeans(n_clusters=3)
# km = KMeans(n_clusters=10)  # for RBG dataset K = 33 
km.fit(X)
km.predict(X)
labels = km.labels_

# print (labels)
# # update labels to original dataframe
# mydataRGB['cluster'] = pd.DataFrame(labels)
# mydataRGB[['file', 'cluster']].head

# Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(X[:,3],X [:,0], c=labels.astype(np.float),s=50)
ax.set_title('R')
ax.set_xlabel('G')
plt.colorbar(scatter)


plt.scatter(X[:, 1], X[:, 0], c=labels.astype(np.float), s=50, cmap='viridis')
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='gray', s=200, alpha=0.5);



# ----------------------------------Hierarchial Clustering ----------------------------------

# Hierarchical clustering for the same dataset
# creating a dataset for hierarchical clustering
dataset2_standardized = X
# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
# some setting for this notebook to actually show the graphs inline
# you probably won't need this
# %matplotlib inline
np.set_printoptions(precision=25, suppress=True)  # suppress scientific float notation
#creating the linkage matrix
H_cluster = linkage(dataset2_standardized,'ward')
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    H_cluster,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=5,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()



# ------------------------------- GMM Clustering models ------------------------------------

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# %matplotlib inline
from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
IX = iris.data

# IX = X

#Gaussian Mixture Model
gmm = GaussianMixture(n_components=5)
gmm.fit(IX)
proba_lists = gmm.predict_proba(IX)

# # convert prob matrices to labels based on index
# gmlabels = proba_lists.argmax(axis=-1)
# # update labels to original dataframe
# mydataRGB['cluster'] = pd.DataFrame(gmlabels)
# mydataRGB[['file', 'cluster']].head


#Plotting
colored_arrays = np.matrix(proba_lists)
colored_tuples = [tuple(i.tolist()[0]) for i in colored_arrays]
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(IX[:, 3], IX[:, 0], IX[:, 2],
          c=colored_tuples, edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("Gaussian Mixture Model", fontsize=14)



##############################################################################################

# ----------------------- KD Tree SIMPLE VERSION---------------------
from sklearn.neighbors import KDTree

tree = KDTree(X)  # metric = 
# metric distance https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
# KDTree.valid_metrics gives list of valid metrics 
# Valid values for metric are: 
# from scikit-learn:
# ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
# from scipy.spatial.distance:
# ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']


# Search in the Kd Tree

# where ft = raw feature 
nx, ny, nz = ft.shape  # know the shape before you flatten
F = ft.reshape ((nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

dist, ind = tree.query(F, k=9)






# ----------------------- KD Tree  RGB ---------------------
# Avg. Time per search: 0.023 s

from sklearn.neighbors import KDTree
import time 


# for feature vector matrices from features daataframe 
XD = list(mydataRGB['imagehist'])
XA = np.asarray(XD)
nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

RGBtree = KDTree(X , metric='manhattan')


# random sample 
q_path = random.sample(imagepaths, 1)[0]

start = time.time()
# test 
ft = ImageSearch_Algo_RGB.RGB_FEATURE (q_path)

# ft = raw feature 
# process 
nx, ny, nz = ft.shape  # know the shape before you flatten
F = ft.reshape ((1, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

dist, ind = RGBtree.query(F, k=50)
t = time.time() - start 

print (ind)
# get the index of searchimage 
print (mydataRGB.index[mydataRGB['file'] == q_path])
print (q_path)

print ( "RGB KDTree Search took ", t, ' secs')
# print the list of files from ind
# mydataRGB.iloc[ ind[0].tolist()]['file']

# Zip results into a list of tuples (score , file) & calculate score 
flist = list (mydataHSV.iloc[ ind[0].tolist()]['file'])
slist = list (dist[0])
result = tuple(zip( slist, flist)) 
a , q = accuracy.accuracy_matches(q_path, result, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )











# --------------------  KD Tree  HSV ---------------------
# Avg. Time per search: 0.033 s


from sklearn.neighbors import KDTree
import time 

YD = list(mydataHSV['imagehist'])
YA = np.asarray(YD)
# nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
# X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

HSVtree = KDTree(YA ) # , metric='euclidean')

#  Example with HSV metrices 
import ImageSearch_Algo_HSV 
import Accuracy as accuracy

# random sample 
q_path = random.sample(imagepaths, 1)[0]

start = time.time()
# test 
fh = ImageSearch_Algo_HSV.HSV_FEATURE (q_path)

fh = np.asarray(fh)
# ft = raw feature 
# process 
nz = fh.shape  # know the shape before you flatten
F = fh.reshape (1, -1) # gives a 2 D matice (sample, value) which can be fed to KMeans 

dist, ind = HSVtree.query(F, k=100)
t = time.time() - start 

print (ind)

# get the index of searchimage 
print (mydataHSV.index[mydataHSV['file'] == q_path])
print (q_path)
print ( "Search took ", t, ' secs')

# Zip results into a list of tuples (score , file) & calculate score 
flist = list (mydataHSV.iloc[ ind[0].tolist()]['file'])
slist = list (dist[0])
result = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 
a , q = accuracy.accuracy_matches(q_path, result, 100)
print ('Accuracy =',  a, '%', '| Quality:', q )