import os
import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths

# for hash all the images in folder / database
import Accuracy as accuracy
import ImageSearch_Algo_Hash
import ImageSearch_Algo_HSV
import ImageSearch_Algo_ORB
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import ImageSearch_Plots as myplots

# # --------------- Reload modules on : 
# %load_ext autoreload
# %autoreload 2

# --------------- VAR COMMONS------------------

IMGDIR = r'./imagesbooks/'
# IMGDIR = r"V:\\Download\\imagesbooks\\"
# IMGDIRPROCESSED = ['']*5
# IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
# IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
# IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
# IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
# IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"

# --------------- TEST PARAMETERS ----------------------#
TESTNAME = "Data519"

# --------------- IMAGES  ----------------------#
imagepaths = list(paths.list_images(IMGDIR))


# ----------- GENERATE ALL FEATURES & SAVE ------------ #

# GEN SIFT 
sift_features_limit = 100
lowe_ratio = 0.7
predictions_count = 50

mydataSIFT, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(
    imagepaths, sift_features_limit)
print("SIFT Feature Generation time :", mytime1)
savefile = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(sift_features_limit)
ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydataSIFT, savefile)
print("SIFT Feature saved to : ", savefile)
# -- END

# GEN ORB 
orb_features_limit = 100

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
mydataHASH, mytime = ImageSearch_Algo_Hash.HASH_GEN(imagepaths, 16)
print("HASH Features Generation time :", mytime)
savefile = 'data/' + TESTNAME + '_PandasDF_HASH_Features'
ImageSearch_Algo_Hash.HASH_SAVE_FEATURES (mydataHASH, savefile)
# -- END


# ----------- GENERATE ALL TREES  ------------ #

# RGB TREE
savefile = 'data/' + TESTNAME + '_RGB_Tree'
myHSVtree = ImageSearch_Algo_RGB.RGB_Create_Tree(mydataRGB, savefile=savefile)

# HSV TREE
savefile = 'data/' + TESTNAME + '_HSV_Tree'
myHSVtree = ImageSearch_Algo_HSV.HSV_Create_Tree(mydataHSV, savefile=savefile)

# HASH TREE
AlgoGenList = ['whash', 'phash', 'dhash', 'ahash'] 
for algo in AlgoGenList : 
    savefile = 'data/' + TESTNAME + '_HASH_Tree_' + str(algo) 
    myHASHTree = ImageSearch_Algo_Hash.HASH_Create_Tree(mydataHASH, savefile=savefile)


# HASH TREE USE HYBRID HASH 
HybridAlgoList = ['whash', 'ahash']
savefile = 'data/' + TESTNAME + '_HASH_Hybrid_Tree_' + str(('_').join (HybridAlgoList)) 
myHybridtree = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, savefile, HybridAlgoList)

# SIFT Tree 
n_clusters = 500
savefile = 'data/' + TESTNAME + '_SIFT_Tree_Cluster' + str(n_clusters)
SIFTtree, SIFTmodel = ImageSearch_Algo_SIFT.SIFT_CREATE_TREE_MODEL(mydataSIFT, savefile, n_clusters)

# ORB Tree



