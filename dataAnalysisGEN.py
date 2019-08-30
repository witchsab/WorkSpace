import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths
from kneed import KneeLocator

import Accuracy as accuracy
import ImageSearch_Algo_Hash
import ImageSearch_Algo_HSV
import ImageSearch_Algo_ORB
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import ImageSearch_Plots as myplots
import Thresholding

# # --------------- Reload modules on :
# %load_ext autoreload
# %autoreload 2


# --------------- TEST PARAMETERS ----------------------#
# TESTNAME = "Data519VERIFY"
# TESTNAME = "Data519_RESIZE320"
# TESTNAME = "DataUKBENCH10K"

# --------------- VAR COMMONS------------------

TESTNAME = "Data519"
IMGDIR = r'./imagesbooks/'

# TESTNAME = "DataUKBENCH10K"
# IMGDIR = r'./ukbench/'

# TESTNAME = "Data519_DENOISE2"
# IMGDIR = r'./images/imagesbooks_DENOISE2/'

# TESTNAME = "Data519_S320"
# IMGDIR = r'./images/imagesbooks_S320/'

# TESTNAME = "Data519_S160"
# IMGDIR = r'./images/imagesbooks_S160/'

# TESTNAME = "Data519_CT2.0"
# IMGDIR = r'./images/imagesbooks_CT2.0/'

# IMGDIR = r"V:\\Download\\imagesbooks\\"
# IMGDIRPROCESSED = ['']*5
# IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
# IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
# IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
# IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
# IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"

# --------------- CONFIG PARAMETERS ----------------------#

ORB_FEATURES_LIMIT = 100
ORB_N_CLUSTERS = 500
SIFT_N_CLUSTERS = 500
SIFT_N_CLUSTERS2 = 50
SIFT_FEATURES_LIMIT = 100
SIFT_FEATURES_LIMIT2 = 300
LOWE_RATIO = 0.7
SIFT_PREDICTIONS_COUNT = 100
RGB_PARAMETERCORRELATIONTHRESHOLD = 0.70 # not needed for generation
kneeHSV = 2
kneeRGB = 2
kneeORB = 2
kneeSIFT = 2
HASHLENGTH = 16

# --------------- IMAGES  ----------------------#
imagepaths =  (list(paths.list_images(IMGDIR)))
myDataFiles = pd.DataFrame( {'file' : imagepaths })




# ----------- GENERATE ALL FEATURES & SAVE ------------ #
print ('Testname: ', TESTNAME)
print ("Generating Features for ", len(imagepaths), "images in", IMGDIR)


# GEN SIFT
sift_features_limit = SIFT_FEATURES_LIMIT
lowe_ratio = LOWE_RATIO
predictions_count = SIFT_PREDICTIONS_COUNT

mydataSIFT, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(
    imagepaths, sift_features_limit)
print("SIFT Feature Generation time :", mytime1)
savefile = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(sift_features_limit)
ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydataSIFT, savefile)
print("SIFT Feature saved to : ", savefile)
# -- END


# GEN SIFT 2 # for TREE kp=300, n_cluster=50 
sift_features_limit = SIFT_FEATURES_LIMIT2
lowe_ratio = LOWE_RATIO
predictions_count = SIFT_PREDICTIONS_COUNT

mydataSIFT2, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(
    imagepaths, sift_features_limit)
print("SIFT Feature Generation time :", mytime1)
savefile = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(sift_features_limit)
ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydataSIFT2, savefile)
print("SIFT Feature saved to : ", savefile)
# -- END



# GEN ORB
orb_features_limit = ORB_FEATURES_LIMIT

mydataORB, mytime1 = ImageSearch_Algo_ORB.GEN_ORB_FEATURES(imagepaths, orb_features_limit)
print("ORB Feature Generation time :", mytime1)
savefile = 'data/' + TESTNAME + '_PandasDF_ORB_Features_kp'+ str(orb_features_limit)
ImageSearch_Algo_ORB.ORB_SAVE_FEATURES (mydataORB, savefile)
print("ORB Feature saved to : ", savefile)
# -- END

# GEN RGB
parametercorrelationthreshold = 0.70 # not needed for generation

mydataRGB, mytime = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print('RGB Feature Generation time', mytime)
savefile = 'data/' + TESTNAME + '_PandasDF_RGB_Features'
ImageSearch_Algo_RGB.RGB_SAVE_FEATURES (mydataRGB, savefile)
print("RGB Feature saved to : ", savefile)
# -- END

# GEN HSV
mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print('HSV Feature Generation time', mytime)
savefile = 'data/' + TESTNAME + '_PandasDF_HSV_Features'
ImageSearch_Algo_HSV.HSV_SAVE_FEATURES (mydataHSV, savefile)
print("HSV Feature saved to : ", savefile)
# -- END


# GEN HASH
mydataHASH, mytime = ImageSearch_Algo_Hash.HASH_GEN(imagepaths, HASHLENGTH)
print("HASH Features Generation time :", mytime)
savefile = 'data/' + TESTNAME + '_PandasDF_HASH_Features'
ImageSearch_Algo_Hash.HASH_SAVE_FEATURES (mydataHASH, savefile)
# -- END

print ("## Feature Generation Complete.")


# ----------- GENERATE ALL TREES  ------------ #

# RGB TREE
savefile = 'data/' + TESTNAME + '_RGB_Tree'
myRGBtree = ImageSearch_Algo_RGB.RGB_Create_Tree(mydataRGB, savefile=savefile)

# HSV TREE
savefile = 'data/' + TESTNAME + '_HSV_Tree'
myHSVtree = ImageSearch_Algo_HSV.HSV_Create_Tree(mydataHSV, savefile=savefile)

# HASH TREE
AlgoGenList = ['whash', 'phash', 'dhash', 'ahash']
for algo in AlgoGenList :
    savefile = 'data/' + TESTNAME + '_HASH_Tree_' + str(algo)
    myHASHTree = ImageSearch_Algo_Hash.HASH_Create_Tree(mydataHASH, savefile=savefile, hashAlgo=algo)

# HASH TREE USE HYBRID HASH
HybridAlgoList = ['whash', 'ahash']
savefile = 'data/' + TESTNAME + '_HASH_Hybrid_Tree_' + str(('_').join (HybridAlgoList))
myHybridtree = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, savefile, HybridAlgoList)


# SIFT FV Tree and Cluster
n_clusters = SIFT_N_CLUSTERS
savefile = 'data/' + TESTNAME + '_SIFT_Tree_Cluster' + str(n_clusters)
mySIFTtree, mySIFTmodel, mySIFTFVHist = ImageSearch_Algo_SIFT.SIFT_CREATE_TREE_MODEL(mydataSIFT, savefile, n_clusters)


# SIFT FV Tree and Cluster with kp=300, n_cluster=50
n_clusters = SIFT_N_CLUSTERS2
savefile = 'data/' + TESTNAME + '_SIFT_Tree_Cluster' + str(n_clusters) + 'kp'+str(SIFT_FEATURES_LIMIT2)
mySIFTtree2, mySIFTmodel2, mySIFTFVHist2 = ImageSearch_Algo_SIFT.SIFT_CREATE_TREE_MODEL(mydataSIFT2, savefile, n_clusters)


# ORB FV Tree and Cluster
n_clusters = ORB_N_CLUSTERS
savefile = 'data/' + TESTNAME + '_ORB_Tree_Cluster' + str(n_clusters)
myORBtree, myORBmodel, myORBFVHist = ImageSearch_Algo_ORB.ORB_CREATE_TREE_MODEL(mydataORB, savefile, n_clusters)

print ("## Tree Generation Complete.")
