import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths
from kneed import KneeLocator

import AccuracyGlobal
import ImageSearch_Algo_Hash
import ImageSearch_Algo_HSV
import ImageSearch_Algo_ORB
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import ImageSearch_Plots as myplots
import Thresholding

# # --------------- Reload modules on :
%load_ext autoreload
%autoreload 2


# --------------- TEST PARAMETERS ----------------------#
# TESTNAME = "Data519_RESIZE320"
TESTNAME = "Data519"

# --------------- VAR COMMONS------------------

IMGDIR = r'./imagesbooks/'
# IMGDIR = r'./images/imagesbooks_DENOISE2/'
# IMGDIR = r'./images/imagesbooks_S160/'
# IMGDIR = r'./images/imagesbooks_S320/'
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
SIFT_FEATURES_LIMIT = 100
LOWE_RATIO = 0.7
SIFT_PREDICTIONS_COUNT = 100
RGB_PARAMETERCORRELATIONTHRESHOLD = 0.70 # not needed for generation
kneeHSV = 2
kneeRGB = 2
kneeORB = 2
kneeSIFT = 2
HASHLENGTH = 16

accuracy = AccuracyGlobal.AccuracyGlobal() # empty class genrated 
accuracy.read(IMGDIR)

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


# ORB FV Tree and Cluster
n_clusters = ORB_N_CLUSTERS
savefile = 'data/' + TESTNAME + '_ORB_Tree_Cluster' + str(n_clusters)
myORBtree, myORBmodel, myORBFVHist = ImageSearch_Algo_ORB.ORB_CREATE_TREE_MODEL(mydataORB, savefile, n_clusters)

print ("## Tree Generation Complete.")

# ----------- GENERATE ALL CLUSTERS  ------------ #

# # # determine n_cluster RGB -> elbow method
# # kneeRGB = ImageSearch_Algo_RGB.RGB_ANALYZE_CLUSTER(mydataRGB, len(mydataRGB.index), int(len(mydataRGB.index)/20))

# # # determine n_cluster HSV -> elbow method
# # kneeHSV = ImageSearch_Algo_HSV.HSV_ANALYZE_CLUSTER (mydataHSV,  len(mydataHSV.index), int(len(mydataHSV.index)/20))

# # ----


# # determine RBG Cluster Size
# # kneeRGB = ImageSearch_Algo_RGB.RGB_ANALYZE_CLUSTER (mydataRGB, 100, 5)
# # create RGB Cluster
# savefile = 'data/' + 'test' + '_RGB_Cluster' + str(kneeRGB)
# RGBClusterModel = ImageSearch_Algo_RGB.RGB_CREATE_CLUSTER (mydataRGB, savefile, n_clusters=150)
# RGBClusterTable = ImageSearch_Algo_RGB.RGB_RUN_CLUSTER(RGBClusterModel, mydataRGB)
# RGBClusterTable.sort_values('file')[['file', 'clusterID']]


# # determine HSV Cluster Size
# # kneeHSV = ImageSearch_Algo_HSV.HSV_ANALYZE_CLUSTER (mydataHSV, 100, 5)
# savefile = 'data/' + 'test' + '_HSV_Cluster' + str(kneeHSV)
# # create RGB Cluster
# HSVClusterModel = ImageSearch_Algo_HSV.HSV_CREATE_CLUSTER (mydataHSV, savefile, n_clusters=200)
# HSVClusterTable = ImageSearch_Algo_HSV.HSV_RUN_CLUSTER(HSVClusterModel, mydataHSV)
# # HSVClusterTable.sort_values('file')[['file', 'clusterID']]


# # determine SIFT FVHist Cluster Size
# # kneeSIFT = ImageSearch_Algo_SIFT.SIFT_ANALYZE_CLUSTER(FVHist, 500, 10)
# # create SIFT Cluster
# savefile = 'data/' + 'test' + '_SIFT_Cluster' + str(kneeSIFT)
# # SIFTClusterModel = ImageSearch_Algo_SIFT.SIFT_CREATE_CLUSTER (mydataSIFT, savefile, n_clusters=kneeSIFT)
# SIFTClusterTable = ImageSearch_Algo_SIFT.SIFT_RUN_CLUSTER (mySIFTFVHist, mydataSIFT, n_clusters=115)
# # SIFTClusterTable.sort_values('file')[['file', 'clusterID']]


# # determine ORB FVHist Cluster Size
# # kneeORB = ImageSearch_Algo_ORB.ORB_ANALYZE_CLUSTER(FVHist, 500, 10)
# # create ORB Cluster
# savefile = 'data/' + 'test' + '_ORB_Cluster' + str(kneeORB)
# # ORBClusterModel = ImageSearch_Algo_ORB.ORB_CREATE_CLUSTER (mydataORB, savefile, n_clusters=kneeORB)
# ORBClusterTable = ImageSearch_Algo_ORB.ORB_RUN_CLUSTER (myORBFVHist, mydataORB, n_clusters=50)
# ORBClusterTable.sort_values('file')[['file', 'clusterID']]


#######################################################################

# -----------  LOAD FEATUTES AND TREES from file  ------------ #

HybridAlgoList = ['whash', 'ahash']
AlgoGenList = ['whash', 'phash', 'dhash', 'ahash'] 

# Files 
file_HASH_Feature = 'data/' + TESTNAME + '_PandasDF_HASH_Features'
file_HASH_HybridTree = 'data/' + TESTNAME + '_HASH_Hybrid_Tree_' + str(('_').join (HybridAlgoList))
file_HSV_Cluster = 'data/' + 'test' + '_HSV_Cluster' + str(kneeHSV)
file_HSV_Feature = 'data/' + TESTNAME + '_PandasDF_HSV_Features'
file_HSV_Tree = 'data/' + TESTNAME + '_HSV_Tree'
file_ORB_Cluster = 'data/' + 'test' + '_ORB_Cluster' + str(kneeORB)
file_ORB_Feature = 'data/' + TESTNAME + '_PandasDF_ORB_Features_kp'+ str(ORB_FEATURES_LIMIT)
file_ORB_TreeCluster = 'data/' + TESTNAME + '_ORB_Tree_Cluster' + str(ORB_N_CLUSTERS)
file_Results = 'data/' + TESTNAME + '_Results'
file_RGB_Cluster = 'data/' + 'test' + '_RGB_Cluster' + str(kneeRGB)
file_RGB_Feature = 'data/' + TESTNAME + '_PandasDF_RGB_Features'
file_RGB_Tree = 'data/' + TESTNAME + '_RGB_Tree'
file_SIFT_Cluster = 'data/' + 'test' + '_SIFT_Cluster' + str(kneeSIFT)
file_SIFT_Feature = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(SIFT_FEATURES_LIMIT)
file_SIFT_TreeCluster = 'data/' + TESTNAME + '_SIFT_Tree_Cluster' + str(SIFT_N_CLUSTERS)

# Features 
mydataRGB = ImageSearch_Algo_RGB.RGB_LOAD_FEATURES (file_RGB_Feature)
mydataHSV = ImageSearch_Algo_HSV.HSV_LOAD_FEATURES (file_HSV_Feature)
mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature)
mydataORB = ImageSearch_Algo_ORB.ORB_LOAD_FEATURES(file_ORB_Feature)
mydataHASH = ImageSearch_Algo_Hash.HASH_LOAD_FEATURES(file_HASH_Feature)

# Tree & Clusters 
myRGBtree = ImageSearch_Algo_RGB.RGB_Load_Tree (file_RGB_Tree)
myHSVtree = ImageSearch_Algo_HSV.HSV_Load_Tree (file_HSV_Tree)
mySIFTtree, mySIFTmodel, mySIFTFVHist = ImageSearch_Algo_SIFT.SIFT_Load_Tree_Model (file_SIFT_TreeCluster)
myORBtree, myORBmodel, myORBFVHist = ImageSearch_Algo_ORB.ORB_Load_Tree_Model(file_ORB_TreeCluster)
myHybridtree =ImageSearch_Algo_Hash.HASH_Load_Tree (file_HASH_HybridTree)

# Hash Algo load all TREES 
myHASH_Trees = {}
for algo in AlgoGenList : 
    savefile = 'data/' + TESTNAME + '_HASH_Tree_' + str(algo)
    myHASHTree = ImageSearch_Algo_Hash.HASH_Load_Tree(savefile)
    myHASH_Trees[algo] = myHASHTree






################################################################################
#                               ALGO CALLS 
################################################################################

# ---------- search HSV Tree
def search_HSV(returnCount=100): 
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE ( myHSVtree, mydataHSV, q_path, returnCount=returnCount)
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

    return imagematcheshsv , searchtimehsv    

# # ---------- search RGB Tree
def search_RGB(returnCount=100, mydataRGB=mydataRGB) : 
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, returnCount=returnCount)
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

    return imagematchesrgb , searchtimergb 

# # ---------- search RGB Correlation
def search_RGB_Corr(returnCount=100, mydataRGB=mydataRGB): 
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

    return imagematchesrgb , searchtimergb 


# # ---------- search SIFT FLANN
def search_SIFT_FLANN(returnCount=100, mydataSIFT=mydataSIFT): 
    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydataSIFT, q_path, sift_features_limit=SIFT_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_sift_Flann'] = a
    row_dict['index_sift_Flann'] = ind
    row_dict['Count_sift_Flann'] = cnt
    row_dict['quality_sift_Flann'] = d
    row_dict['time_sift_Flann'] = searchtimesift

    return imagepredictions, searchtimesift



# # ---------- search SIFT BF
def search_SIFT_BF(returnCount=100, mydataSIFT=mydataSIFT): 
    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydataSIFT, q_path, sift_features_limit=SIFT_FEATURES_LIMIT, lowe_ratio=LOWE_RATIO, predictions_count=returnCount)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_sift_BF'] = a
    row_dict['index_sift_BF'] = ind
    row_dict['Count_sift_BF'] = cnt
    row_dict['quality_sift_BF'] = d
    row_dict['time_sift_BF'] = searchtimesift

    return imagepredictions, searchtimesift



# # ---------- search SIFT BOVW Tree
def search_SIFT_BOVW(returnCount=100): 
    imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_TREE(q_path, mySIFTmodel, mySIFTtree, mydataSIFT, returnCount=returnCount, kp=100)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_sift_tree'] = a
    row_dict['index_sift_tree'] = ind
    row_dict['Count_sift_tree'] = cnt
    row_dict['quality_sift_tree'] = d
    row_dict['time_sift_tree'] = searchtime

    return imagematches, searchtime



# # ---------- search ORB FLANN-LSH 
def search_ORB_FLANN(returnCount=100, mydataORB=mydataORB) : 
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_FLANN(mydataORB, q_path, ORB_features_limit=ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount )
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_Flann'] = a
    row_dict['index_orb_Flann'] = ind
    row_dict['Count_orb_Flann'] = cnt
    row_dict['quality_orb_Flann'] = d
    row_dict['time_orb_Flann'] = searchtime

    return imagematches, searchtime


# # ---------- search ORB BF
def search_ORB_BF(returnCount=100, mydataORB=mydataORB) : 
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_BF(mydataORB, q_path, ORB_features_limit=ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount )
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_BF'] = a
    row_dict['index_orb_BF'] = ind
    row_dict['Count_orb_BF'] = cnt
    row_dict['quality_orb_BF'] = d
    row_dict['time_orb_BF'] = searchtime

    return imagematches, searchtime


# # ---------- search ORB BF NEW
def search_ORB_BF2(returnCount=100, mydataORB=mydataORB) :
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_MODBF(mydataORB, q_path, ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount )
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_BF2'] = a
    row_dict['index_orb_BF2'] = ind
    row_dict['Count_orb_BF2'] = cnt
    row_dict['quality_orb_BF2'] = d
    row_dict['time_orb_BF2'] = searchtime

    return imagematches, searchtime


# # ---------- search ORB BOVW Tree
def search_ORB_BOVW (returnCount=100) : 
    imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_TREE(q_path, myORBmodel, myORBtree, mydataORB, returnCount=100, kp=ORB_FEATURES_LIMIT)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_orb_tree'] = a
    row_dict['index_orb_tree'] = ind
    row_dict['Count_orb_tree'] = cnt
    row_dict['quality_orb_tree'] = d
    row_dict['time_orb_tree'] = searchtime
    
    return imagematches, searchtime


# # ---------- search HASH All
def search_HASH_All(returnCount=100): 
    # AlgoGenList = ['whash', 'phash', 'dhash', 'ahash']    
    for algo in AlgoGenList :
        imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_TREE(myHASH_Trees[algo], mydataHASH, q_path, hashAlgo=algo, hashsize=16, returnCount=returnCount)
        a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
        # print ('Accuracy =',  a, '%', '| Quality:', d )
        # print ('Count', cnt, ' | position', ind)
        row_dict['acc_HASH_'+str(algo)] = a
        row_dict['index_HASH_'+str(algo)] = ind
        row_dict['Count_HASH_'+str(algo)] = cnt
        row_dict['quality_HASH_'+str(algo)] = d
        row_dict['time_HASH_'+str(algo)] = searchtime


# # ---------- search HASH specific Algo 
def search_HASH( algo='whash', returnCount=100): 
    # AlgoGenList = ['whash', 'phash', 'dhash', 'ahash'] 
    imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_TREE(myHASH_Trees[algo], mydataHASH, q_path,hashAlgo=algo, hashsize=16, returnCount=returnCount)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_HASH_'+str(algo)] = a
    row_dict['index_HASH_'+str(algo)] = ind
    row_dict['Count_HASH_'+str(algo)] = cnt
    row_dict['quality_HASH_'+str(algo)] = d
    row_dict['time_HASH_'+str(algo)] = searchtime

    return imagematches, searchtime


# # ---------- search Hybrid HASH
def search_HASH_HYBRID (returnCount=100): 
    # HybridAlgoList = ['whash', 'ahash']
    imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE( myHybridtree, mydataHASH, q_path,hashAlgoList=HybridAlgoList, hashsize=16, returnCount=returnCount)
    a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', ind)
    row_dict['acc_HASH_Hybrid'] = a
    row_dict['index_HASH_Hybrid'] = ind
    row_dict['Count_HASH_Hybrid'] = cnt
    row_dict['quality_HASH_Hybrid'] = d
    row_dict['time_HASH_Hybrid'] = searchtime

    return imagematches, searchtime


# ---------- Algo A = ( HSV(100) + RGB (100) => SIFT BF )
def search_AlgoA ( candidates=100, verbose=False ): 
    toplist = []
    start = time.time()
    # run RGB
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, returnCount=candidates)
    # run HSV
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE ( myHSVtree, mydataHSV, q_path, returnCount=candidates)
    # create shortlist for SIFT 
    filteredSIFTData = Thresholding.filter_sift_candidates( [imagematcheshsv, imagematchesrgb], mydataSIFT)
    # run SIFT 
    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(filteredSIFTData, q_path, sift_features_limit=100 , lowe_ratio=LOWE_RATIO, predictions_count=SIFT_PREDICTIONS_COUNT)

    # threshold RGB 
    final_RGB_List = Thresholding.autothreshold_knee(imagematchesrgb)

    # thresold HSV 
    final_HSV_List = Thresholding.autothreshold_knee(imagematcheshsv)

    # MERGE OPEARATION FROM ALL RESULTS 
    # SIFT LIST 
    final_SIFT_List = Thresholding.imagepredictions_to_list(imagepredictions)
    
    # merge lists of algo results: HSV Thresh, RGB Thresh, SIFT
    toplist = Thresholding.merge_results([final_HSV_List, final_RGB_List, final_SIFT_List], False)
    
    # find accuracy and append to dict 
    a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, toplist, 20 )
    t = time.time() - start
    row_dict['acc_algo_A'] = a
    row_dict['index_algo_A'] = ind
    row_dict['Count_algo_A'] = cnt
    row_dict['quality_algo_A'] = d
    row_dict['time_algo_A'] = t

    # run if verbose enabled; DEBUGGING
    if verbose: 

        print ('index FINAL AlgoA: ', ind)
        # append SIFT Results 
        a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
        print ('SIFT-A Accuracy =',  a, '%', '| Quality:', d )
        print ('SIFT-A Count', cnt, ' | position', ind)
        row_dict['acc_Algo_A_SIFT'] = a
        row_dict['index_Algo_A_SIFT'] = ind
        row_dict['Count_Algo_A_SIFT'] = cnt
        row_dict['quality_Algo_A_SIFT'] = d
        # row_dict['time_Algo_A_SIFT'] = searchtime
        # get current accurracy of RGB     
        a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
        print ('index RGB   : ', ind)
        # get thresholded accurracy of RGB     
        a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, final_RGB_List, 20 )
        print ('index RGB Th:', ind)
        # update candidates RGB
        row_dict['index_Algo_A_cRGB'] = ind
        # get current accurracy for HSV
        a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
        print ('index HSV   : ', ind)
        # get thresholded accurracy for HSV 
        a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, final_HSV_List, 20 )
        print ('index HSV Th: ', ind)
        # update candidates 
        row_dict['index_Algo_A_cHSV'] = ind


    return imagepredictions, t


def algo_selector( algo, return_count = 100):
    if algo == "search_AlgoA": # Tree
        imagepredictions, searchtime = search_AlgoA (candidates=return_count, verbose=False)
    elif algo == "search_HASH": # Tree
        imagepredictions, searchtime = search_HASH (returnCount=return_count) 
    elif algo == "search_HASH_HYBRID": # Tree
        imagepredictions, searchtime = search_HASH_HYBRID (returnCount=return_count)
    elif algo == "search_ORB_BF": # Full
        imagepredictions, searchtime = search_ORB_BF(returnCount=return_count)
    elif algo == "search_ORB_BF2": # Full
        imagepredictions, searchtime = search_ORB_BF2(returnCount=return_count) 
    elif algo == "search_ORB_BOVW": # Tree
        imagepredictions, searchtime = search_ORB_BOVW (returnCount=return_count) 
    elif algo == "search_ORB_FLANN": # Full
        imagepredictions, searchtime = search_ORB_FLANN(returnCount=return_count) 
    elif algo == "search_RGB": # Tree
        imagepredictions, searchtime = search_RGB(returnCount=return_count) 
    elif algo == "search_RGB_Corr": # Full
        imagepredictions, searchtime = search_RGB_Corr(returnCount=return_count)
    elif algo == "search_SIFT_BF": # Full
        imagepredictions, searchtime = search_SIFT_BF(returnCount=return_count) 
    elif algo == "search_SIFT_BOVW": # Tree
        imagepredictions, searchtime = search_SIFT_BOVW(returnCount=return_count)
    elif algo == "search_SIFT_FLANN": # Full
        imagepredictions, searchtime = search_SIFT_FLANN(returnCount=return_count)
    elif algo == "search_HSV": # Tree
        imagepredictions, searchtime = search_HSV(returnCount=return_count)
    else : 
        print ("No Algo Found")
        return 0,0 
    return (imagepredictions, searchtime)


def algo_selector_final( algo, algoFrame,  return_count = 100):
    
    if algo == "search_ORB_BF": # Full
        imagepredictions, searchtime = search_ORB_BF(returnCount=return_count, mydataORB=algoFrame)
    elif algo == "search_ORB_BF2": # Full
        imagepredictions, searchtime = search_ORB_BF2(returnCount=return_count, mydataORB=algoFrame) 
    elif algo == "search_ORB_FLANN": # Full
        imagepredictions, searchtime = search_ORB_FLANN(returnCount=return_count, mydataORB=algoFrame) 
    elif algo == "search_RGB_Corr": # Full
        imagepredictions, searchtime = search_RGB_Corr(returnCount=return_count, mydataRGB=algoFrame)
    elif algo == "search_SIFT_BF": # Full
        imagepredictions, searchtime = search_SIFT_BF(returnCount=return_count,mydataSIFT=algoFrame) 
    elif algo == "search_SIFT_FLANN": # Full
        imagepredictions, searchtime = search_SIFT_FLANN(returnCount=return_count,mydataSIFT=algoFrame)
    else : 
        print ("No Algo Found")
        return 0,0 
    return (imagepredictions, searchtime)


def algomixerAppend (algos, return_count, algoname='NewAlgo') : 
    # myAlgos = [ search_RGB, search_SIFT_BF ]
    # algomixerFunnel (myAlgos)
    start = time.time()

    algoResults = []
    algoTimes = []
    for algo in algos: 
        thisResult, thisTime = algo_selector (algo, return_count=return_count)
        algoResults.append(thisResult)
        algoTimes.append(thisTime)
    
    # generate algo uniques: apply threshold for each Result (imagematches)     
    unique_final_list = []
    for result in algoResults : 
        unique_final_list.append(Thresholding.autothreshold_knee(result))

    # MERGE OPEARATION FROM ALL RESULTS 
    # algo uniques + final algo detections results 

    toplist = unique_final_list.copy()

    # retaining all the individual results as well (P.S: note difference from algomixerFunnel)
    for result in algoResults :    
        toplist.append(Thresholding.imagepredictions_to_list(result))

    # merge lists of algo results and remove duplicates order[[commons], [algo1], [algo2]...]
    toplist = Thresholding.merge_results( toplist, False)

    t = time.time() - start

    # find accuracy and append to dict 
    a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, toplist, 20 )
    print ('index F_'+algoname+': ', ind)

    row_dict['acc_'+ algoname] = a
    row_dict['index_'+ algoname] = ind
    row_dict['Count_'+ algoname] = cnt
    row_dict['quality_'+ algoname] = d
    row_dict['time_'+ algoname] = t


def algomixerFunnel (algos, return_count, finalalgo, finalalgoDataframe, algoname='NewAlgo') : 
    # myAlgos = [ search_RGB, search_SIFT_BF ]
    # algomixerFunnel (myAlgos)
    start = time.time()

    algoResults = []
    algoTimes = []
    for algo in algos: 
        algoResult, algoTime = algo_selector (algo, return_count=return_count)
        algoResults.append(algoResult)
        algoTimes.append(algoTime)

    # generate candidates (short listing)
    filteredFeatureData = Thresholding.filter_candidates( algoResults, finalalgoDataframe)
    # run Final Algo (detection) 
    imagepredictions,searchtimesift = algo_selector_final(finalalgo,filteredFeatureData,return_count=return_count)

    # generate algo uniques: apply threshold for each Result (imagematches)     
    unique_final_list = []
    for Result in algoResults : 
        unique_final_list.append(Thresholding.autothreshold_knee(Result))

    # print (unique_final_list)
    # MERGE OPEARATION FROM ALL RESULTS 
    # algo uniques + final algo detections results 
    final_algo_List = Thresholding.imagepredictions_to_list(imagepredictions)
    
    toplist = unique_final_list.copy()
    # copy all commons from candidates threshold list to front (2 lists)
    toplist = [Thresholding.merge_results( toplist, False)]

    # Add final algo derivatives to toplist 
    toplist.append(final_algo_List)
    # merge lists of algo results: HSV Thresh, RGB Thresh, SIFT
    toplist = Thresholding.merge_results( toplist, False)

    t = time.time() - start

    # find accuracy and append to dict 
    a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, toplist, 20 )
    print ('index F_'+algoname+': ', ind)
    row_dict['acc_'+ algoname] = a
    row_dict['index_'+ algoname] = ind
    row_dict['Count_'+ algoname] = cnt
    row_dict['quality_'+ algoname] = d
    row_dict['time_'+ algoname] = t


#######################################################################
# -----------  DATA COLLECTION START    ------------ #


# image dirs path 
imagepaths = (list(paths.list_images(IMGDIR)))


# ********************** ALL INDIVIDUAL ALGO DATA ********************* #
gt = accuracy.check_ground_truth()
# def runBenchmark50(count=8) : 
# initialize 
Results = pd.DataFrame(columns=['file'])
# iterate over all samples: 
for q_path in imagepaths[:count]: 

    # initialize locals  
    row_dict = {'file':q_path } 

    search_HSV()
    search_RGB() 
    search_RGB_Corr() 

    search_SIFT_BF()
    search_SIFT_FLANN()
    search_SIFT_BOVW()

    search_ORB_FLANN()
    search_ORB_BF()
    search_ORB_BF2()
    search_ORB_BOVW()   
    
    search_HASH_All()
    search_HASH_HYBRID()

    # Funnel algo 
    search_AlgoA(100, False)

    # --------- Append Results to Results
    Results = Results.append( row_dict , ignore_index=True)
    print ( 'Completed ', imagepaths.index(q_path), q_path)

# ---------- SAVE ALL FILES TO DISK
# Save Frame to csv 
Results.to_csv( 'data/' + TESTNAME + '_RESULTS_50Bench_testGlobal.csv')
print ("Data Collection Completed ")

# Save Frame to pickle
savefile = 'data/' + TESTNAME + '_Results_50Bench_testGlobal'
outfile = open (savefile + '.pickle', 'wb')
pickle.dump( Results, outfile )
# ---------- SAVED

# runBenchmark50()  # test bench 


# *************************  CUSTOM ALGO DATA **************************** #
# AlgoMixerAppend : 
#   runs specified algos by text calls and merges result
# AlgoMixerFunnel : 
#   runs specified algosm generates candidates and then runs final algo
#   thresholds candidates for top list 
#   merges all the finals to results


# initialize 
Results = pd.DataFrame(columns=['file'])
# for q_path in imagepaths[30:35]: 
# for q_path in imagepaths[100:201]: 
for q_path in imagepaths: 
    row_dict = {'file':q_path } 

    # ------------Generic Algo Full Sample 
    search_HSV()
    search_RGB() 
    search_RGB_Corr() 

    search_SIFT_BF()
    search_SIFT_FLANN()
    search_SIFT_BOVW()

    search_ORB_FLANN()
    search_ORB_BF()
    search_ORB_BF2()
    search_ORB_BOVW()   
    
    search_HASH_All()
    search_HASH_HYBRID()

    # ---------- ALGO SELECTOR 
    # algo_selector('search_HSV', 100)
    # search_RGB()

    # ----------- generate custom combination algos w/ adaptive thresholding
    algomixerAppend(['search_HSV', 'search_RGB'], 100, 'AlgoS')
    algomixerAppend(['search_ORB_BOVW', 'search_SIFT_BOVW'], 100, 'AlgoF')

    # ----------- generate custom Funnel algos
    algomixerFunnel(['search_HSV', 'search_RGB'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoA')
    # algomixerFunnel(['search_HSV', 'search_RGB'], 100, 'search_SIFT_FLANN', mydataSIFT, 'F_SIFT2')
    algomixerFunnel(['search_HSV', 'search_RGB', 'search_SIFT_BOVW'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoB')
    algomixerFunnel(['search_HSV', 'search_RGB', 'search_SIFT_BOVW'], 50, 'search_SIFT_BF', mydataSIFT, 'AlgoBA')
    algomixerFunnel(['search_HSV', 'search_RGB', 'search_ORB_BOVW'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoC')

    Results = Results.append( row_dict , ignore_index=True)
    print ( 'Completed ', imagepaths.index(q_path), q_path)

# ---------- SAVE ALL FILES TO DISK
# Save Frame to csv 
Results.to_csv( 'data/' + TESTNAME + '_RESULTS_mix100_ALL.csv')
print ("Data Collection Completed ")

# Save Frame to pickle
savefile = 'data/' + TESTNAME + '_Results_mix100_ALL' # + str(int(time.time())) 
outfile = open (savefile + '.pickle', 'wb')
pickle.dump( Results, outfile )
# ---------- SAVED





# ## EXAMPLE of single candidate generator --------
# # initialize 
# Results = pd.DataFrame(columns=['file'])
# for q_path in imagepaths[30:35]: 
# # for q_path in imagepaths[100:201]: 
# # for q_path in imagepaths: 
#     row_dict = {'file':q_path } 

#     algomixerFunnel(['search_SIFT_BOVW'], 200, 'search_SIFT_BF', mydataSIFT, 'AlgoBA')
#     algomixerFunnel(['search_ORB_BOVW'], 200, 'search_SIFT_BF', mydataSIFT, 'AlgoC')

#     Results = Results.append( row_dict , ignore_index=True)
#     print ( 'Completed ', imagepaths.index(q_path), q_path)

# # ---------- SAVE ALL FILES TO DISK
# # Save Frame to csv 
# Results.to_csv( 'data/' + TESTNAME + '_RESULTS_mix100_1P.csv')
# print ("Data Collection Completed ")

# # Save Frame to pickle
# savefile = 'data/' + TESTNAME + '_Results_mix100_1P' # + str(int(time.time())) 
# outfile = open (savefile + '.pickle', 'wb')
# pickle.dump( Results, outfile )
# # ---------- SAVED



# --------------------- HASH CUSTOM TEST 
# image dirs path 
imagepaths = sorted(list(paths.list_images(IMGDIR)))
Results = pd.DataFrame(columns=['file'])
# iterate over all samples: 
for q_path in imagepaths: 

    # initialize locals  
    row_dict = {'file':q_path } 
    search_HASH_All()
    search_HASH_HYBRID()
    # --------- Append Results to Results
    Results = Results.append( row_dict , ignore_index=True)
    print ( 'Completed ', imagepaths.index(q_path), q_path)

# SAVE ALL FILES TO DISK
# Save Frame to csv 
Results.to_csv( 'data/' + TESTNAME + '_RESULTS_HASH_Bench_GlobalTest.csv')
print ("Data Collection Completed ")

# Save Frame to pickle
savefile = 'data/' + TESTNAME + '_Results_HASH_Bench_Global'
outfile = open (savefile + '.pickle', 'wb')
pickle.dump( Results, outfile )
# ---------- SAVED --- HASH END
