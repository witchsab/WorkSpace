import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths

import Accuracy as accuracy
import ImageSearch_Algo_Hash
import ImageSearch_Algo_HSV
import ImageSearch_Algo_ORB
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import ImageSearch_Plots as myplots

# # --------------- Reload modules on :
%load_ext autoreload
%autoreload 2


# --------------- TEST PARAMETERS ----------------------#
TESTNAME = "Data519"

# --------------- VAR COMMONS------------------

IMGDIR = r'./imagesbooks/'
# IMGDIR = r"V:\\Download\\imagesbooks\\"
# IMGDIRPROCESSED = ['']*5
# IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
# IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
# IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
# IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
# IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"

# --------------- CONFIG PARAMETERS ----------------------#

ORB_FEATURES_LIMIT = 200
ORB_N_CLUSTERS = 500
SIFT_N_CLUSTERS = 500
SIFT_FEATURES_LIMIT = 100
LOWE_RATIO = 0.7
SIFT_PREDICTIONS_COUNT = 100
RGB_PARAMETERCORRELATIONTHRESHOLD = 0.70 # not needed for generation

# --------------- IMAGES  ----------------------#
imagepaths = sorted (list(paths.list_images(IMGDIR)))
myDataFiles = pd.DataFrame( {'file' : imagepaths })




# ----------- GENERATE ALL FEATURES & SAVE ------------ #

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
mydataHASH, mytime = ImageSearch_Algo_Hash.HASH_GEN(imagepaths, 16)
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
    myHASHTree = ImageSearch_Algo_Hash.HASH_Create_Tree(mydataHASH, savefile=savefile)

# HASH TREE USE HYBRID HASH
HybridAlgoList = ['whash', 'ahash']
savefile = 'data/' + TESTNAME + '_HASH_Hybrid_Tree_' + str(('_').join (HybridAlgoList))
myHybridtree = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, savefile, HybridAlgoList)


# SIFT FV Tree and Cluster
n_clusters = SIFT_N_CLUSTERS
savefile = 'data/' + TESTNAME + '_SIFT_Tree_Cluster' + str(n_clusters)
SIFTtree, SIFTmodel, SIFTFVHist = ImageSearch_Algo_SIFT.SIFT_CREATE_TREE_MODEL(mydataSIFT, savefile, n_clusters)


# ORB FV Tree and Cluster
n_clusters = ORB_N_CLUSTERS
savefile = 'data/' + TESTNAME + '_ORB_Tree_Cluster' + str(n_clusters)
ORBtree, ORBmodel, ORBFVHist = ImageSearch_Algo_ORB.ORB_CREATE_TREE_MODEL(mydataORB, savefile, n_clusters)

print ("## Tree Generation Complete.")

# ----------- GENERATE ALL CLUSTERS  ------------ #

# # determine n_cluster RGB -> elbow method
# kneeRGB = ImageSearch_Algo_RGB.RGB_ANALYZE_CLUSTER(mydataRGB, len(mydataRGB.index), int(len(mydataRGB.index)/20))

# # determine n_cluster HSV -> elbow method
# kneeHSV = ImageSearch_Algo_HSV.HSV_ANALYZE_CLUSTER (mydataHSV,  len(mydataHSV.index), int(len(mydataHSV.index)/20))

# ----


# determine RBG Cluster Size
# kneeRGB = ImageSearch_Algo_RGB.RGB_ANALYZE_CLUSTER (mydataRGB, 100, 5)
# create RGB Cluster
savefile = 'data/' + 'test' + '_RGB_Cluster' + str(kneeRGB)
RGBClusterModel = ImageSearch_Algo_RGB.RGB_CREATE_CLUSTER (mydataRGB, savefile, n_clusters=150)
RGBClusterTable = ImageSearch_Algo_RGB.RGB_RUN_CLUSTER(RGBClusterModel, mydataRGB)
RGBClusterTable.sort_values('file')[['file', 'clusterID']]


# determine HSV Cluster Size
# kneeHSV = ImageSearch_Algo_HSV.HSV_ANALYZE_CLUSTER (mydataHSV, 100, 5)
savefile = 'data/' + 'test' + '_HSV_Cluster' + str(kneeHSV)
# create RGB Cluster
HSVClusterModel = ImageSearch_Algo_HSV.HSV_CREATE_CLUSTER (mydataHSV, savefile, n_clusters=200)
HSVClusterTable = ImageSearch_Algo_HSV.HSV_RUN_CLUSTER(HSVClusterModel, mydataHSV)
# HSVClusterTable.sort_values('file')[['file', 'clusterID']]


# determine SIFT FVHist Cluster Size
# kneeSIFT = ImageSearch_Algo_SIFT.SIFT_ANALYZE_CLUSTER(FVHist, 500, 10)
# create SIFT Cluster
savefile = 'data/' + 'test' + '_SIFT_Cluster' + str(kneeSIFT)
# SIFTClusterModel = ImageSearch_Algo_SIFT.SIFT_CREATE_CLUSTER (mydataSIFT, savefile, n_clusters=kneeSIFT)
SIFTClusterTable = ImageSearch_Algo_SIFT.SIFT_RUN_CLUSTER (SIFTFVHist, mydataSIFT, n_clusters=115)
# SIFTClusterTable.sort_values('file')[['file', 'clusterID']]


# determine ORB FVHist Cluster Size
# kneeORB = ImageSearch_Algo_ORB.ORB_ANALYZE_CLUSTER(FVHist, 500, 10)
# create ORB Cluster
savefile = 'data/' + 'test' + '_ORB_Cluster' + str(kneeORB)
# ORBClusterModel = ImageSearch_Algo_ORB.ORB_CREATE_CLUSTER (mydataORB, savefile, n_clusters=kneeORB)
ORBClusterTable = ImageSearch_Algo_ORB.ORB_RUN_CLUSTER (ORBFVHist, mydataORB, n_clusters=50)
ORBClusterTable.sort_values('file')[['file', 'clusterID']]


#######################################################################
# -----------  DATA COLLECTION START    ------------ #


# initialize 
Results = pd.DataFrame(columns=['file'])

# load all TREES 
AlgoGenList = ['whash', 'phash', 'dhash', 'ahash']    
hashAlgoDict = {}
for algo in AlgoGenList : 
    savefile = 'data/' + TESTNAME + '_HASH_Tree_' + str(algo)
    myHASHTree = ImageSearch_Algo_Hash.HASH_Load_Tree(savefile)
    hashAlgoDict[algo] = myHASHTree
    

imagepaths = list(paths.list_images(IMGDIR))

# iterate over all samples: 
for q_path in imagepaths[:80]: 

    # initialize locals  
    toplist = []
    row_dict = {'file':q_path }   


    start = time.time()

    # ---------- search HSV Tree
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE ( myHSVtree, mydataHSV, q_path, returnCount=100)
    a, d, i_hsv, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
    row_dict['acc_hsv'] = a
    row_dict['index_hsv'] = i_hsv
    row_dict['Count_hsv'] = cnt
    row_dict['quality_hsv'] = d
    row_dict['time_hsv'] = searchtimehsv
    # print ('HSV Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', i_hsv)
    # x = autothreshold (imagematcheshsv)
    # toplist = toplist + x
    # # print (x)

    # ---------- search RGB Tree
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, returnCount=100)
    # y= autothreshold (imagematchesrgb)
    # toplist = toplist + y
    # print (y)
    a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
    row_dict['acc_rgb'] = a
    row_dict['index_rgb'] = ind
    row_dict['Count_rgb'] = cnt
    row_dict['quality_rgb'] = d
    row_dict['time_rgb'] = searchtimergb
    # print ('RGB Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)


    # ---------- search RGB Correlation
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, correl_threshold=RGB_PARAMETERCORRELATIONTHRESHOLD)
    # y= autothreshold (imagematchesrgb)
    # toplist = toplist + y
    # print (y)
    a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
    row_dict['acc_rgb_corr'] = a
    row_dict['index_rgb_corr'] = ind
    row_dict['Count_rgb_corr'] = cnt
    row_dict['quality_rgb_corr'] = d
    row_dict['time_rgb_corr'] = searchtimergb
    # print ('RGB Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)


    # ---------- search SIFT FLANN
    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydataSIFT, q_path, sift_features_limit=100 , lowe_ratio=LOWE_RATIO, predictions_count=SIFT_PREDICTIONS_COUNT)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_sift_Flann'] = a
    row_dict['index_sift_Flann'] = ind
    row_dict['Count_sift_Flann'] = cnt
    row_dict['quality_sift_Flann'] = d
    row_dict['time_sift_Flann'] = searchtimesift


    # ---------- search SIFT BF
    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydataSIFT, q_path, sift_features_limit=100 , lowe_ratio=LOWE_RATIO, predictions_count=SIFT_PREDICTIONS_COUNT)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_sift_BF'] = a
    row_dict['index_sift_BF'] = ind
    row_dict['Count_sift_BF'] = cnt
    row_dict['quality_sift_BF'] = d
    row_dict['time_sift_BF'] = searchtimesift


    # ---------- search SIFT BOVW Tree
    imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_TREE(q_path, SIFTmodel, SIFTtree, mydataSIFT, returnCount=100, kp=100)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_sift_tree'] = a
    row_dict['index_sift_tree'] = ind
    row_dict['Count_sift_tree'] = cnt
    row_dict['quality_sift_tree'] = d
    row_dict['time_sift_tree'] = searchtime


    # ---------- search ORB FLANN-LSH 
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_FLANN(mydataORB, q_path, ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=SIFT_PREDICTIONS_COUNT )
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_Flann'] = a
    row_dict['index_orb_Flann'] = ind
    row_dict['Count_orb_Flann'] = cnt
    row_dict['quality_orb_Flann'] = d
    row_dict['time_orb_Flann'] = searchtime


    # ---------- search ORB BF
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_BF(mydataORB, q_path, ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=SIFT_PREDICTIONS_COUNT )
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_BF'] = a
    row_dict['index_orb_BF'] = ind
    row_dict['Count_orb_BF'] = cnt
    row_dict['quality_orb_BF'] = d
    row_dict['time_orb_BF'] = searchtime


    # ---------- search ORB BF NEW
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_MODBF(mydataORB, q_path, ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=SIFT_PREDICTIONS_COUNT )
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_BF2'] = a
    row_dict['index_orb_BF2'] = ind
    row_dict['Count_orb_BF2'] = cnt
    row_dict['quality_orb_BF2'] = d
    row_dict['time_orb_BF2'] = searchtime


    # ---------- search ORB BOVW Tree
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_TREE(q_path, ORBmodel, ORBtree, mydataORB, returnCount=100, kp=ORB_FEATURES_LIMIT)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_tree'] = a
    row_dict['index_orb_tree'] = ind
    row_dict['Count_orb_tree'] = cnt
    row_dict['quality_orb_tree'] = d
    row_dict['time_orb_tree'] = searchtime


    # ---------- search HASH All
    AlgoGenList = ['whash', 'phash', 'dhash', 'ahash']    
    for algo in AlgoGenList :
        imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_TREE(hashAlgoDict[algo], mydataHASH, q_path,hashAlgo=algo, hashsize=16, returnCount=100)
        a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
        # print ('Accuracy =',  a, '%', '| Quality:', d )
        # print ('Count', cnt, ' | position', ind)
        row_dict['acc_HASH_'+str(algo)] = a
        row_dict['index_HASH_'+str(algo)] = ind
        row_dict['Count_HASH_'+str(algo)] = cnt
        row_dict['quality_HASH_'+str(algo)] = d
        row_dict['time_HASH_'+str(algo)] = searchtime
    

    # ---------- search Hybrid HASH
    HybridAlgoList = ['whash', 'ahash']
    imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE( myHybridtree, mydataHASH, q_path,hashAlgoList=HybridAlgoList, hashsize=16, returnCount=100)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_HASH_Hybrid'] = a
    row_dict['index_HASH_Hybrid'] = ind
    row_dict['Count_HASH_Hybrid'] = cnt
    row_dict['quality_HASH_Hybrid'] = d
    row_dict['time_HASH_Hybrid'] = searchtime


    # --------- Append Results to Results
    Results = Results.append( row_dict , ignore_index=True)
    print ( 'Completed ', imagepaths.index(q_path), q_path)


Results.to_csv( 'data/' + TESTNAME + '_RESULTS.csv')
print ("Data Collection Completed ")