# HSV Functions

import os
import pickle
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


def HSV_GEN (listOfImagePaths): 

    Trainhist = pd.DataFrame(columns=['file','imagehist'])

    start = time.time()    
    
    # initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))

    # Iterate through all images in paths 
    for f in listOfImagePaths:
        image = cv2.imread(f)
        # describe the image
        features = cd.describe(image)
        Trainhist =  Trainhist.append({'file':f,'imagehist':features}, ignore_index=True)
    
    t= time.time() - start
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(custompaths), t))
    # print (Trainhist.head())
    return (Trainhist, t)

def HSV_FEATURE (searchimagepath): 

    # initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))

    queryImage = cv2.imread(searchimagepath)

    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))
    queryFeatures = cd.describe(queryImage)

    return queryFeatures


'''
Save Pandas dataframe to pickle 
Datafram format : file , imagehist
'''
def HSV_SAVE_FEATURES ( mydataHSV, savefile='testHSV_Data') : 
    
    # save the tree #example # treeName = 'testRGB_Data.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( mydataHSV, outfile)

    
'''
Load Pandas dataframe from pickle 
Datafram format : file , imagehist
'''
def HSV_LOAD_FEATURES ( openfile='testHSV_Data') : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    mydataHSV = pickle.load(infile)
    infile.close()

    return mydataHSV



def HSV_Create_Tree ( mydataHSV, savefile='testHSV'  ) : 
    
    YD = list(mydataHSV['imagehist'])
    YA = np.asarray(YD)
    # nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
    # X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    HSVtree = KDTree(YA ) # , metric='euclidean')
    
    # save the tree #example # treeName = 'testHSV.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump(HSVtree,outfile)

    return HSVtree



def HSV_Load_Tree ( openfile='testHSV'  ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    HSVTree = pickle.load(infile)
    infile.close()

    return HSVTree


def HSV_CREATE_CLUSTER ( mydataHSV, savefile='testHSVcluster', n_clusters=50 ) : 
    
    YD = list(mydataHSV['imagehist'])
    X = np.asarray(YD)

    HSVCluster = KMeans(n_clusters=n_clusters)
    HSVCluster.fit(X)
    # HSVCluster.predict(X)
    # labels = HSVCluster.labels_
    # # print (labels)

    # # update labels to original dataframe
    # mydataHSV['clusterID'] = pd.DataFrame(labels)
    
    # save the tree #example # treeName = 'testHSV.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump(HSVCluster,outfile)

    return HSVCluster

def HSV_RUN_CLUSTER ( HSVCluster, mydataHSV ) : 
    YD = list(mydataHSV['imagehist'])
    X = np.asarray(YD)
    HSVCluster.predict(X)
    labels = HSVCluster.labels_
    # print (labels)

    # update labels to original dataframe
    mydataHSV['clusterID'] = pd.DataFrame(labels)

    return mydataHSV
    

def HSV_ANALYZE_CLUSTER (mydataHSV, n_clusters=200, step=10) :  
    print ('n_clusters:', n_clusters, '| step:',  step)
    YD = list(mydataHSV['imagehist'])
    X = np.asarray(YD)
    
    # Calculate clusters using Elbow criteria 
    wcss = []
    for i in range(1, n_clusters, step ):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, n_clusters, step), wcss)
    # plt.plot(range(1, 11), elbowIndex)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # find elbow using Kneelocator package 
    # https://github.com/arvkevi/kneed#find-knee
    
    elbow = KneeLocator( list(range(1,n_clusters, step)), wcss, S=1.0, curve='convex', direction='decreasing')
    print ('Detected Elbow cluster value :', elbow.knee)

    return elbow.knee


def HSV_LOAD_CLUSTER ( openfile='testHSVcluster'  ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    HSVCluster = pickle.load(infile)
    infile.close()

    return HSVCluster


'''
Params: 
HSVTree = Tree object 
mydataHSV = pandas dataframe of the tree (same order no filter or change)
searchimagepath = string path of the search image 

Output: 
list of tuples: [(score, matchedfilepath) ]
time = total searching time 
'''
def HSV_SEARCH_TREE ( HSVtree , mydataHSV,  searchimagepath, returnCount=100): 

    start = time.time()
    
    # get the feature from the input image 
    fh = HSV_FEATURE (searchimagepath)

    fh = np.asarray(fh)
    # ft = raw feature 
    # process 
    nz = fh.shape  # know the shape before you flatten
    F = fh.reshape (1, -1) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    # the search; k = number of returns expected 
    # returns distances, index of the items in the order of the input tree nparray 
    dist, ind = HSVtree.query(F, k=returnCount)
    t = time.time() - start 

    flist = list (mydataHSV.iloc[ ind[0].tolist()]['file'])
    slist = list (dist[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 

    return (matches, t)



def HSV_SEARCH (feature, searchimagepath): 

    start = time.time()    

    queryImage = cv2.imread(searchimagepath)

    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))
    queryFeatures = cd.describe(queryImage)

    matches = []

    for index, row in feature.iterrows():
        distance = chi2_distance( row['imagehist'], queryFeatures)
        # distance = cv2.compareHist(queryFeatures, row['imagehist'], cv2.HISTCMP_CORREL)
        matches.append((distance, row['file']))

    matches.sort(key=lambda x : x[0] , reverse = False)

    t= time.time() - start
    return (matches, t)



def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


# ------------------- class color descriptor for images -------------------

class ColorDescriptor:
    def __init__(self, bins):
    # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]

        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
            [0, 180, 0, 256, 0, 256])

        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()

        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist
