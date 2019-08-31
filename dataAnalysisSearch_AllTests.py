import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths
from kneed import KneeLocator

import AccuracyGlobal
# import Accuracy as accuracy

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
# TESTNAME = "Data520_RESIZE320"
# TESTNAME = "Data520VERIFY"
# TESTNAME = "Data520"
# TESTNAME = "DataUKBENCH10K"

# --------------- VAR COMMONS------------------

# TESTNAME = "Data520"
# IMGDIR = r'./imagesbooks/'

# TESTNAME = "DataUKBENCH10K"
# IMGDIR = r'./ukbench/'

# TESTNAME = "Data520_DENOISE2"
# IMGDIR = r'./images/imagesbooks_DENOISE2/'

# TESTNAME = "Data520_S320"
# IMGDIR = r'./images/imagesbooks_S320/'

# TESTNAME = "Data520_S160"
# IMGDIR = r'./images/imagesbooks_S160/'

# TESTNAME = "Data520_CT2.0"
# IMGDIR = r'./images/imagesbooks_CT2.0/'

# TESTNAME = "Data520AUG"
# IMGDIR = r'./images/imagesbooks_AUG/'

# IMGDIR = r"V:\\Download\\imagesbooks\\"
# IMGDIRPROCESSED = ['']*5
# IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
# IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
# IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
# IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
# IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"

def SEARCH_TESTS(TESTNAME, IMGDIR):

    # --------------- CONFIG PARAMETERS ----------------------#

    ORB_FEATURES_LIMIT = 100
    ORB_FEATURES_LIMIT2 = 500   
    ORB_N_CLUSTERS = 500
    ORB_N_CLUSTERS2 = 50    # 500 # (option2)

    SIFT_FEATURES_LIMIT = 100
    SIFT_FEATURES_LIMIT2 = 300
    SIFT_N_CLUSTERS = 500
    SIFT_N_CLUSTERS2 = 50   # 100 # (option2)

    LOWE_RATIO = 0.7
    SIFT_PREDICTIONS_COUNT = 100
    RGB_PARAMETERCORRELATIONTHRESHOLD = 0.70 # not needed for generation
    kneeHSV = 2
    kneeRGB = 2
    kneeORB = 2
    kneeSIFT = 2
    HASHLENGTH = 16
    ACCURRACY_RANGE = 20

    accuracy = AccuracyGlobal.AccuracyGlobal() # empty class genrated 
    accuracy.read(IMGDIR)



    # --------------- IMAGES  ----------------------#
    imagepaths =  (list(paths.list_images(IMGDIR)))
    myDataFiles = pd.DataFrame( {'file' : imagepaths })

    print ('Testname: ', TESTNAME)
    print ("Searching Features for ", len(imagepaths), "images in", IMGDIR)



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

    file_SIFT_Feature2 = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(SIFT_FEATURES_LIMIT2)
    file_SIFT_TreeCluster2 = 'data/' + TESTNAME + '_SIFT_Tree_Cluster' + str(SIFT_N_CLUSTERS2) + 'kp'+str(SIFT_FEATURES_LIMIT2)

    file_ORB_Feature2 = 'data/' + TESTNAME + '_PandasDF_ORB_Features_kp'+ str(ORB_FEATURES_LIMIT2)
    file_ORB_TreeCluster2 = 'data/' + TESTNAME + '_ORB_Tree_Cluster' + str(ORB_N_CLUSTERS2) + 'kp'+str(ORB_FEATURES_LIMIT2)



    # Features 
    mydataRGB = ImageSearch_Algo_RGB.RGB_LOAD_FEATURES (file_RGB_Feature)
    mydataHSV = ImageSearch_Algo_HSV.HSV_LOAD_FEATURES (file_HSV_Feature)
    mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature)
    mydataSIFT2 = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature2)
    mydataORB = ImageSearch_Algo_ORB.ORB_LOAD_FEATURES(file_ORB_Feature)
    mydataORB2 = ImageSearch_Algo_ORB.ORB_LOAD_FEATURES(file_ORB_Feature2)
    mydataHASH = ImageSearch_Algo_Hash.HASH_LOAD_FEATURES(file_HASH_Feature)

    # Tree & Clusters 
    myRGBtree = ImageSearch_Algo_RGB.RGB_Load_Tree (file_RGB_Tree)
    myHSVtree = ImageSearch_Algo_HSV.HSV_Load_Tree (file_HSV_Tree)
    mySIFTtree, mySIFTmodel, mySIFTFVHist = ImageSearch_Algo_SIFT.SIFT_Load_Tree_Model (file_SIFT_TreeCluster)
    mySIFTtree2, mySIFTmodel2, mySIFTFVHist2 = ImageSearch_Algo_SIFT.SIFT_Load_Tree_Model (file_SIFT_TreeCluster2)
    myORBtree, myORBmodel, myORBFVHist = ImageSearch_Algo_ORB.ORB_Load_Tree_Model(file_ORB_TreeCluster)
    myORBtree2, myORBmodel2, myORBFVHist2 = ImageSearch_Algo_ORB.ORB_Load_Tree_Model(file_ORB_TreeCluster2)
    myHybridtree =ImageSearch_Algo_Hash.HASH_Load_Tree (file_HASH_HybridTree)

    # Hash Algo load all TREES 
    myHASH_Trees = {}
    for algo in AlgoGenList : 
        savefile = 'data/' + TESTNAME + '_HASH_Tree_' + str(algo)
        myHASHTree = ImageSearch_Algo_Hash.HASH_Load_Tree(savefile)
        myHASH_Trees[algo] = myHASHTree






    ################################################################################
    #                               ALGO CALLS                                     #
    ################################################################################

    # ---------- search HSV Tree
    def search_HSV(returnCount=100, write=False): 
        imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE ( myHSVtree, mydataHSV, q_path, returnCount=returnCount)
        if write: 
            a, d, i_hsv, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, ACCURRACY_RANGE)
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
    def search_RGB(returnCount=100, mydataRGB=mydataRGB, write=False) : 
        imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, returnCount=returnCount)
        # y= autothreshold (imagematchesrgb)
        # toplist = toplist + y
        # print (y)
        if write: 
            a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, ACCURRACY_RANGE)
            row_dict['acc_rgb'] = a
            row_dict['index_rgb'] = ind
            row_dict['Count_rgb'] = cnt
            row_dict['quality_rgb'] = d
            row_dict['time_rgb'] = searchtimergb
            # print ('RGB Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)

        return imagematchesrgb , searchtimergb 

    # # ---------- search RGB Correlation
    def search_RGB_Corr(returnCount=100, mydataRGB=mydataRGB, write=False): 
        imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, correl_threshold=RGB_PARAMETERCORRELATIONTHRESHOLD)
        # y= autothreshold (imagematchesrgb)
        # toplist = toplist + y
        # print (y)
        if write: 
            a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, ACCURRACY_RANGE)
            row_dict['acc_rgb_corr'] = a
            row_dict['index_rgb_corr'] = ind
            row_dict['Count_rgb_corr'] = cnt
            row_dict['quality_rgb_corr'] = d
            row_dict['time_rgb_corr'] = searchtimergb
            # print ('RGB Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)

        return imagematchesrgb , searchtimergb 


    # # ---------- search SIFT FLANN
    def search_SIFT_FLANN(returnCount=100, mydataSIFT=mydataSIFT, write=False): 
        imagepredictionsFLANN , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydataSIFT, q_path, sift_features_limit=SIFT_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount)
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictionsFLANN, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_sift_Flann'] = a
            row_dict['index_sift_Flann'] = ind
            row_dict['Count_sift_Flann'] = cnt
            row_dict['quality_sift_Flann'] = d
            row_dict['time_sift_Flann'] = searchtimesift

        return imagepredictionsFLANN, searchtimesift



    # # ---------- search SIFT BF
    def search_SIFT_BF(returnCount=100, mydataSIFT=mydataSIFT, txt='', write=False): 
        imagepredictionsBF , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydataSIFT, q_path, sift_features_limit=SIFT_FEATURES_LIMIT, lowe_ratio=LOWE_RATIO, predictions_count=returnCount)
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictionsBF, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_sift_BF'+txt] = a
            row_dict['index_sift_BF'+txt] = ind
            row_dict['Count_sift_BF'+txt] = cnt
            row_dict['quality_sift_BF'+txt] = d
            row_dict['time_sift_BF'+txt] = searchtimesift

        return imagepredictionsBF, searchtimesift



    # # ---------- search SIFT BOVW Tree
    def search_SIFT_BOVW(returnCount=100, mySIFTmodel=mySIFTmodel, mySIFTtree=mySIFTtree, mydataSIFT=mydataSIFT, txt='', write=False): 
        imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_TREE(q_path, mySIFTmodel, mySIFTtree, mydataSIFT, returnCount=returnCount, kp=100)
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_sift_tree'+txt] = a
            row_dict['index_sift_tree'+txt] = ind
            row_dict['Count_sift_tree'+txt] = cnt
            row_dict['quality_sift_tree'+txt] = d
            row_dict['time_sift_tree'+txt] = searchtime

        return imagematches, searchtime



    # # ---------- search ORB FLANN-LSH 
    def search_ORB_FLANN(returnCount=100, mydataORB=mydataORB, txt='', write=False) : 
        imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_FLANN(mydataORB, q_path, ORB_features_limit=ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount )
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_orb_Flann' + txt] = a
            row_dict['index_orb_Flann' + txt] = ind
            row_dict['Count_orb_Flann' + txt] = cnt
            row_dict['quality_orb_Flann' + txt] = d
            row_dict['time_orb_Flann' + txt] = searchtime

        return imagematches, searchtime


    # # ---------- search ORB BF
    def search_ORB_BF(returnCount=100, mydataORB=mydataORB, write=False) : 
        imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_BF(mydataORB, q_path, ORB_features_limit=ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount )
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_orb_BF'] = a
            row_dict['index_orb_BF'] = ind
            row_dict['Count_orb_BF'] = cnt
            row_dict['quality_orb_BF'] = d
            row_dict['time_orb_BF'] = searchtime

        return imagematches, searchtime


    # # ---------- search ORB BF NEW
    def search_ORB_BF2(returnCount=100, mydataORB=mydataORB, write=False) :
        imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_MODBF(mydataORB, q_path, ORB_FEATURES_LIMIT , lowe_ratio=LOWE_RATIO, predictions_count=returnCount )
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_orb_BF2'] = a
            row_dict['index_orb_BF2'] = ind
            row_dict['Count_orb_BF2'] = cnt
            row_dict['quality_orb_BF2'] = d
            row_dict['time_orb_BF2'] = searchtime

        return imagematches, searchtime


    # # ---------- search ORB BOVW Tree
    def search_ORB_BOVW (returnCount=100, myORBmodel=myORBmodel, myORBtree=myORBtree, mydataORB=mydataORB, txt='', write=False) : 
        imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_TREE(q_path, myORBmodel, myORBtree, mydataORB, returnCount=100, kp=ORB_FEATURES_LIMIT)
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_orb_tree'+ txt] = a
            row_dict['index_orb_tree'+ txt] = ind
            row_dict['Count_orb_tree'+ txt] = cnt
            row_dict['quality_orb_tree'+ txt] = d
            row_dict['time_orb_tree'+ txt] = searchtime
        
        return imagematches, searchtime


    # # ---------- search HASH All
    def search_HASH_All(returnCount=100, write=False): 
        # AlgoGenList = ['whash', 'phash', 'dhash', 'ahash']    
        for algo in AlgoGenList :
            imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_TREE(myHASH_Trees[algo], mydataHASH, q_path, hashAlgo=algo, hashsize=16, returnCount=returnCount)
            if write: 
                a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
                # print ('Accuracy =',  a, '%', '| Quality:', d )
                # print ('Count', cnt, ' | position', ind)
                row_dict['acc_HASH_'+str(algo)] = a
                row_dict['index_HASH_'+str(algo)] = ind
                row_dict['Count_HASH_'+str(algo)] = cnt
                row_dict['quality_HASH_'+str(algo)] = d
                row_dict['time_HASH_'+str(algo)] = searchtime


    # # ---------- search HASH specific Algo 
    def search_HASH( algo='whash', returnCount=100, write=False): 
        # AlgoGenList = ['whash', 'phash', 'dhash', 'ahash'] 
        imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_TREE(myHASH_Trees[algo], mydataHASH, q_path,hashAlgo=algo, hashsize=16, returnCount=returnCount)
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
            # print ('Accuracy =',  a, '%', '| Quality:', d )
            # print ('Count', cnt, ' | position', ind)
            row_dict['acc_HASH_'+str(algo)] = a
            row_dict['index_HASH_'+str(algo)] = ind
            row_dict['Count_HASH_'+str(algo)] = cnt
            row_dict['quality_HASH_'+str(algo)] = d
            row_dict['time_HASH_'+str(algo)] = searchtime

        return imagematches, searchtime


    # # ---------- search Hybrid HASH
    def search_HASH_HYBRID (returnCount=100, write=False): 
        # HybridAlgoList = ['whash', 'ahash']
        imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE( myHybridtree, mydataHASH, q_path,hashAlgoList=HybridAlgoList, hashsize=16, returnCount=returnCount)
        if write: 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, ACCURRACY_RANGE)
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
        a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, toplist, ACCURRACY_RANGE)
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
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, ACCURRACY_RANGE)
            print ('SIFT-A Accuracy =',  a, '%', '| Quality:', d )
            print ('SIFT-A Count', cnt, ' | position', ind)
            row_dict['acc_Algo_A_SIFT'] = a
            row_dict['index_Algo_A_SIFT'] = ind
            row_dict['Count_Algo_A_SIFT'] = cnt
            row_dict['quality_Algo_A_SIFT'] = d
            # row_dict['time_Algo_A_SIFT'] = searchtime
            # get current accurracy of RGB     
            a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, ACCURRACY_RANGE)
            print ('index RGB   : ', ind)
            # get thresholded accurracy of RGB     
            a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, final_RGB_List, ACCURRACY_RANGE)
            print ('index RGB Th:', ind)
            # update candidates RGB
            row_dict['index_Algo_A_cRGB'] = ind
            # get current accurracy for HSV
            a, d, ind, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, ACCURRACY_RANGE)
            print ('index HSV   : ', ind)
            # get thresholded accurracy for HSV 
            a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, final_HSV_List, ACCURRACY_RANGE)
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
        elif algo == "search_ORB_BOVW2": # Tree
            imagepredictions, searchtime = search_ORB_BOVW (returnCount=return_count, myORBmodel=myORBmodel2, myORBtree=myORBtree2, mydataORB=mydataORB2, txt='2')     
        elif algo == "search_ORB_FLANN": # Full
            imagepredictions, searchtime = search_ORB_FLANN(returnCount=return_count) 
        elif algo == "search_ORB_FLANN2": # Full
            imagepredictions, searchtime = search_ORB_FLANN(returnCount=return_count, mydataORB=mydataORB2, txt='2') 
        elif algo == "search_RGB": # Tree
            imagepredictions, searchtime = search_RGB(returnCount=return_count) 
        elif algo == "search_RGB_Corr": # Full
            imagepredictions, searchtime = search_RGB_Corr(returnCount=return_count)
        elif algo == "search_SIFT_BF": # Full
            imagepredictions, searchtime = search_SIFT_BF(returnCount=return_count) 
        elif algo == "search_SIFT_BF2": # Full
            imagepredictions, searchtime = search_SIFT_BF(returnCount=return_count, mydataSIFT=mydataSIFT2, txt='2') 
        elif algo == "search_SIFT_BOVW": # Tree
            imagepredictions, searchtime = search_SIFT_BOVW(returnCount=return_count)
        elif algo == "search_SIFT_BOVW2": # Tree
            imagepredictions, searchtime = search_SIFT_BOVW(returnCount=return_count, mySIFTmodel=mySIFTmodel2, mySIFTtree=mySIFTtree2, mydataSIFT=mydataSIFT2, txt='2')
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
        a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, toplist, ACCURRACY_RANGE)
        print ('index F_'+algoname+': ', ind)

        row_dict['acc_'+ algoname] = a
        row_dict['index_'+ algoname] = ind
        row_dict['Count_'+ algoname] = cnt
        row_dict['quality_'+ algoname] = d
        row_dict['time_'+ algoname] = t


    def algomixerFunnel (algos, return_count, finalalgo, finalalgoDataframe, algoname='NewAlgo', write=False) : 
        '''
        algos [list]: list of candidare algos
        return_count (int) : number of candidates to be generated 
        finalalgo (str) : list of candidare algos
        finalalgoDataframe (pd.Dataframe): dataframe of the finalAlgo to be filtered and used 
        algoname (str)  : 'column name of datframe for the final reported accuracy, time etc.
        write (bool): True / False -> whether to report funnel accurracy, time before merge with thresholded candidates 
        '''

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
        if write: 
            # find accuracy and append to dict 
            a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagepredictions, ACCURRACY_RANGE)
            t = time.time() - start
            row_dict['acc_'+ algoname + '_BM'] = a
            row_dict['index_'+ algoname + '_BM'] = ind
            row_dict['Count_'+ algoname + '_BM'] = cnt
            row_dict['quality_'+ algoname + '_BM'] = d
            row_dict['time_'+ algoname + '_BM'] = t

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
        a ,d, ind, cnt = accuracy.accuracy_from_list(q_path, toplist, ACCURRACY_RANGE)
        print ('index F_'+algoname+': ', ind)
        row_dict['acc_'+ algoname] = a
        row_dict['index_'+ algoname] = ind
        row_dict['Count_'+ algoname] = cnt
        row_dict['quality_'+ algoname] = d
        row_dict['time_'+ algoname] = t


    #######################################################################
    # -----------  DATA COLLECTION START    ------------ #

    gt = accuracy.check_ground_truth()

    # image dirs path 
    imagepaths = (list(paths.list_images(IMGDIR)))

    # imagepaths = ['./imagesbooks/ukbench09622.jpg','./imagesbooks/ukbench10066.jpg','./imagesbooks/ukbench03864.jpg','./imagesbooks/ukbench06696.jpg','./imagesbooks/ukbench08546.jpg','./imagesbooks/ukbench05988.jpg','./imagesbooks/ukbench02718.jpg','./imagesbooks/ukbench05945.jpg','./imagesbooks/ukbench05779.jpg','./imagesbooks/ukbench08054.jpg','./imagesbooks/ukbench10166.jpg','./imagesbooks/ukbench05776.jpg','./imagesbooks/ukbench03865.jpg','./imagesbooks/ukbench06004.jpg','./imagesbooks/ukbench08048.jpg','./imagesbooks/ukbench05874.jpg','./imagesbooks/ukbench03098.jpg','./imagesbooks/ukbench05600.jpg','./imagesbooks/ukbench06047.jpg','./imagesbooks/ukbench10065.jpg']


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
        search_HSV(write=True)
        search_RGB(write=True) 
        # search_RGB_Corr(write=True) 

        search_SIFT_BF(txt='100', write=True)
        # search_SIFT_BF(mydataSIFT=mydataSIFT2, txt='300', write=True)
        # search_SIFT_FLANN(write=True)
        search_SIFT_BOVW(write=True)
        search_SIFT_BOVW(mySIFTmodel=mySIFTmodel2, mySIFTtree=mySIFTtree2, mydataSIFT=mydataSIFT2, txt='kp300n50', write=True)
        
        # search_ORB_FLANN(txt='100', write=True)
        # search_ORB_FLANN(mydataORB=mydataORB2, txt='500', write=True)
        # search_ORB_BF(write=True)
        # search_ORB_BF2(write=True)
        # search_ORB_BOVW(write=True)
        # search_ORB_BOVW(myORBmodel=myORBmodel2, myORBtree=myORBtree2, mydataORB=mydataORB2, txt='kp500n50',write=True)
        
        search_HASH_All(write=True)
        search_HASH_HYBRID(write=True)

        # ---------- ALGO SELECTOR 
        # algo_selector('search_HSV', 100)
        # search_RGB()

        # ----------- generate custom combination algos w/ adaptive thresholding
        # algomixerAppend(['search_HSV', 'search_RGB'], 100, 'AlgoS')
        # algomixerAppend(['search_ORB_BOVW', 'search_SIFT_BOVW'], 100, 'AlgoF')
        # algomixerAppend(['search_ORB_BOVW2', 'search_SIFT_BOVW2'], 100, 'AlgoFX')
        algomixerAppend(['search_HSV', 'search_RGB', 'search_SIFT_BOVW2'], 100, 'AlgoX')

        # ----------- generate custom Funnel algos
        
        algomixerFunnel(['search_HSV', 'search_RGB'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoA', write=True)
        # algomixerFunnel(['search_HSV', 'search_RGB'], 100, 'search_SIFT_BF', mydataSIFT2, 'AlgoA2', write=True)
        # algomixerFunnel(['search_HSV', 'search_RGB'], 100, 'search_SIFT_FLANN', mydataSIFT, 'F_SIFT2')

        algomixerFunnel(['search_HSV', 'search_RGB', 'search_SIFT_BOVW'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoB_100', write=True)
        algomixerFunnel(['search_HSV', 'search_RGB', 'search_SIFT_BOVW'], 50, 'search_SIFT_BF', mydataSIFT, 'AlgoB_50', write=True)

        # algomixerFunnel(['search_HSV', 'search_RGB', 'search_ORB_BOVW'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoC', write=True)
        # algomixerFunnel(['search_HSV', 'search_RGB', 'search_ORB_BOVW2'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoC2', write=True)

        algomixerFunnel(['search_HSV', 'search_RGB', 'search_SIFT_BOVW2'], 100, 'search_SIFT_BF', mydataSIFT, 'AlgoB2_100', write=True)
        algomixerFunnel(['search_HSV', 'search_RGB', 'search_SIFT_BOVW2'], 50, 'search_SIFT_BF', mydataSIFT, 'AlgoB2_50', write=True)


        Results = Results.append( row_dict , ignore_index=True)
        print ( 'Completed ', imagepaths.index(q_path), q_path)

    # ---------- SAVE ALL FILES TO DISK
    # Save Frame to csv 
    Results.to_csv( 'data/' + TESTNAME + '_RESULTS_FINAL.csv')
    print ("Data Collection Completed ")

    # Save Frame to pickle
    savefile = 'data/' + TESTNAME + '_RESULTS_FINAL' # + str(int(time.time())) 
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( Results, outfile )
    # ---------- SAVED

    # Calculate statistics and save to a file 
    stats = Results.describe()
    stats.to_csv( 'data/' + TESTNAME + '_RESULTS_FINAL_stats.csv')

    print ("Data Analysis Completed.")


# Define all TESTNAMES AND DIRS 

TESTS = [
    ("Data520",             r'./imagesbooks/'),
    ("Data520_DENOISE2",    r'./images/imagesbooks_DENOISE2/'   ),
    ("Data520_S320",        r'./images/imagesbooks_S320/'       ),
    ("Data520_S160",        r'./images/imagesbooks_S160/'       ),
    ("Data520_R90",         r'./images/imagesbooks_R90/'        ),
    ("Data520_CT2.0",       r'./images/imagesbooks_CT2.0/'      ),
    ("Data520_EQ2",         r'./images/imagesbooks_EQ2/'        ),
    # ("DataUKBENCH10K",      r'./ukbench/'),
    # ("Data520AUG",          r'./images/imagesbooks_AUG/'),
]

# Run all the tests

for testname, imgdir in TESTS: 
    SEARCH_TESTS(testname, imgdir)