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

# --------------- Reload modules on :
%load_ext autoreload
%autoreload 2


# --------------- TEST PARAMETERS ----------------------#
# TESTNAME = "Data519_RESIZE320"
TESTNAME = "Data519_KP_nCluster"

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

# --------------- IMAGES  ----------------------#
imagepaths =  (list(paths.list_images(IMGDIR)))
myDataFiles = pd.DataFrame( {'file' : imagepaths })




for kp in [100, 200, 300, 400, 500]:


    # ----------- GENERATE ALL FEATURES & SAVE ------------ #

    # GEN SIFT
    sift_features_limit = kp
    lowe_ratio = LOWE_RATIO
    predictions_count = SIFT_PREDICTIONS_COUNT

    mydataSIFT, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(
        imagepaths, sift_features_limit)
    print("SIFT Feature Generation time :", mytime1)
    savefile = 'opt/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(sift_features_limit)
    ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydataSIFT, savefile)
    print("SIFT Feature saved to : ", savefile)
    # -- END

    # # GEN ORB
    # orb_features_limit = ORB_FEATURES_LIMIT

    # mydataORB, mytime1 = ImageSearch_Algo_ORB.GEN_ORB_FEATURES(imagepaths, orb_features_limit)
    # print("ORB Feature Generation time :", mytime1)
    # savefile = 'opt/' + TESTNAME + '_PandasDF_ORB_Features_kp'+ str(orb_features_limit)
    # ImageSearch_Algo_ORB.ORB_SAVE_FEATURES (mydataORB, savefile)
    # print("ORB Feature saved to : ", savefile)
    # # -- END
    

    print("## Feature Generation Complete.")


    # ----------- GENERATE ALL TREES  ------------ #
    print("## Tree Generation Started for kp", kp)

    for n in [50, 100, 500, 1000, 2000, 3000, 4000, 5000] :
        # SIFT FV Tree and Cluster
        n_clusters = n

        # ORB FV Tree and Cluster
        savefile = 'opt/' + TESTNAME + '_ORB_Tree_kp'+str(kp)+'Cluster' + str(n_clusters)
        myORBtree, myORBmodel, myORBFVHist = ImageSearch_Algo_ORB.ORB_CREATE_TREE_MODEL(mydataORB, savefile, n_clusters)

        print("## ORB Tree completed for kp", kp, "n_cluster", n_clusters)


    print("## Tree Generation Complete for kp", kp)

# -----------  LOAD FEATUTES, TREES from file and RUN  ------------ #

# Initialize Stats 
ResultStats = pd.DataFrame()

for kp in [100, 200, 300, 400, 500]:
    for n in [50, 100, 500, 1000, 2000, 3000, 4000, 5000] :

        n_clusters = n 
        
        # # Files
        # file_SIFT_Cluster = 'opt/' + 'test' + '_SIFT_Cluster' + str(kneeSIFT)
        file_SIFT_Feature = 'opt/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(kp)
        file_SIFT_TreeCluster = 'opt/' + TESTNAME + '_SIFT_Tree_kp'+str(kp)+'Cluster' + str(n_clusters)


        # file_ORB_Cluster = 'opt/' + 'test' + '_ORB_Cluster' + str(kneeORB)
        # file_ORB_Feature = 'opt/' + TESTNAME + '_PandasDF_ORB_Features_kp'+ str(ORB_FEATURES_LIMIT)
        # file_ORB_TreeCluster = 'opt/' + TESTNAME + '_ORB_Tree_Cluster' + str(ORB_N_CLUSTERS)
        # # Features
        mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature)
        # mydataORB = ImageSearch_Algo_ORB.ORB_LOAD_FEATURES(file_ORB_Feature)

        # # Tree & Clusters
        mySIFTtree, mySIFTmodel, mySIFTFVHist = ImageSearch_Algo_SIFT.SIFT_Load_Tree_Model (file_SIFT_TreeCluster)
        # myORBtree, myORBmodel, myORBFVHist = ImageSearch_Algo_ORB.ORB_Load_Tree_Model(file_ORB_TreeCluster)


        ################################################################################
        #                               ALGO CALLS                                     #
        ################################################################################

        # # ---------- search SIFT BOVW Tree
        def search_SIFT_BOVW(returnCount=100, write=False):
            imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_TREE_SELF(q_path, mySIFTmodel, mySIFTtree, mydataSIFT, returnCount=returnCount, kp=kp)
            if write:
                a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
                # print('Accuracy =',  a, '%', '| Quality:', d )
                # print('Count', cnt, ' | position', ind)
                row_dict['acc_sift_tree'] = a
                row_dict['index_sift_tree'] = ind
                row_dict['Count_sift_tree'] = cnt
                row_dict['quality_sift_tree'] = d
                row_dict['time_sift_tree'] = searchtime

            return imagematches, searchtime


        # # # ---------- search ORB BOVW Tree
        # def search_ORB_BOVW(returnCount=100, write=False):
        #     imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_TREE(q_path, myORBmodel, myORBtree, mydataORB, returnCount=100, kp=ORB_FEATURES_LIMIT)
        #     if write:
        #         a ,d, ind, cnt = accuracy.accuracy_matches(q_path, imagematches, 20 )
        #         # print('Accuracy =',  a, '%', '| Quality:', d )
        #         # print('Count', cnt, ' | position', ind)
        #         row_dict['acc_orb_tree'] = a
        #         row_dict['index_orb_tree'] = ind
        #         row_dict['Count_orb_tree'] = cnt
        #         row_dict['quality_orb_tree'] = d
        #         row_dict['time_orb_tree'] = searchtime
        
        #     return imagematches, searchtime

        #######################################################################
        # -----------  DATA COLLECTION START    ------------ #


        # # image dirs path
        # imagepaths = (list(paths.list_images(IMGDIR)))


        # initialize
        Results = pd.DataFrame(columns=['file'])
        # for q_path in imagepaths[30:35]:
        # for q_path in imagepaths[100:201]:
        start = time.time()
        sample = 100
        for q_path in imagepaths[:sample]:
            row_dict = {'file':q_path}

            # ------------Generic Algo Full Sample
            
            search_SIFT_BOVW(write=True)

            # search_ORB_BOVW(write=True)  
        
            Results = Results.append( row_dict, ignore_index=True)
            # print( 'Completed ', imagepaths.index(q_path), q_path)

        # ---------- SAVE ALL FILES TO DISK
        # Save Frame to csv
        Results.to_csv( 'opt/' + TESTNAME + '_RESULTS_kp'+str(kp)+'Cluster'+ str(n_clusters)+'.csv')
        # print('Data Collection Completed  kp'+str(kp)+' Cluster' + str(n_clusters))

        print ('kp:%5d' %(kp), 'n:%5d t: %2.4f' %(n, (time.time() - start)/sample), "\tACC: %2.2f" %(Results['acc_sift_tree'].mean()), "COUNT: %2.2f" %(Results['Count_sift_tree'].mean()) )

        ResultStats = ResultStats.append ({'kp': kp, 'n_cluster': n, 'Acc': Results['acc_sift_tree'].mean(), 'Count': Results['Count_sift_tree'].mean(), 'Acc_std': Results['acc_sift_tree'].std(), 'Count_std': Results['Count_sift_tree'].std(), 'Time': Results['time_sift_tree'].mean()}, ignore_index=True)
        # # Save Frame to pickle
        # savefile = 'opt/' + TESTNAME + '_Results' # + str(int(time.time()))
        # outfile = open (savefile + '.pickle', 'wb')
        # pickle.dump( Results, outfile )
        # # ---------- SAVED

ResultStats.to_csv( 'opt/' + TESTNAME + '_RESULTS_kp_vs_nCluster.csv')
