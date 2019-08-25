import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths

import Accuracy as accuracy
import ImageSearch_Algo_SIFT

# # --------------- Reload modules on :
%load_ext autoreload
%autoreload 2


# --------------- TEST PARAMETERS ----------------------#
# TESTNAME = "Data519_RESIZE320"
TESTNAME = "siftdata"

# --------------- VAR COMMONS------------------

IMGDIR = r'./imagesbooks/'


# --------------- CONFIG PARAMETERS ----------------------#


SIFT_FEATURES_LIMIT = 100
LOWE_RATIO = 0.7
SIFT_PREDICTIONS_COUNT = 100


# --------------- IMAGES  ----------------------#
imagepaths = sorted (list(paths.list_images(IMGDIR)))
myDataFiles = pd.DataFrame( {'file' : imagepaths })

# ----------- GENERATE ALL FEATURES & SAVE ------------ #
# kps = [50]
kps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# GEN SIFT
for kp in kps:
    sift_features_limit = kp
    lowe_ratio = LOWE_RATIO
    predictions_count = SIFT_PREDICTIONS_COUNT

    mydataSIFT, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(
        imagepaths, sift_features_limit)
    print("SIFT Feature Generation time :", mytime1)
    savefile = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(sift_features_limit)
    ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydataSIFT, savefile)
    print("SIFT Feature saved to : ", savefile)
    # -- END




# # ---------- search SIFT BF
def search_SIFT_BF(returnCount=100, mydataSIFT=mydataSIFT,SIFT_FEATURES_LIMIT=SIFT_FEATURES_LIMIT): 
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




# ********************** ALL INDIVIDUAL ALGO DATA ********************* #
# kps = [50]
kps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
count = 50
mysiftanalysis = pd.DataFrame()
# GEN SIFT
for kp in kps:
    # load feature
    file_SIFT_Feature = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp'+ str(kp)
    mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature)

    # initialize 
    Results = pd.DataFrame(columns=['file'])
    # iterate over all samples: 
    for q_path in imagepaths[:count]: 

        # initialize locals  
        row_dict = {'file':q_path }     
        search_SIFT_BF(returnCount = 100, mydataSIFT=mydataSIFT,SIFT_FEATURES_LIMIT=SIFT_FEATURES_LIMIT )
        # search_SIFT_FLANN()
        # search_SIFT_BOVW()
        # --------- Append Results to Results
        Results = Results.append( row_dict , ignore_index=True)
        print ( 'Completed ', imagepaths.index(q_path), q_path)


    # ---------- SAVE ALL FILES TO DISK
    # Save Frame to csv 
    Results.to_csv( 'data/' + TESTNAME + '_RESULTS_SIFT_kp'+str(kp)+'.csv')
    print ("Data Collection Completed ")

    
    m1 = Results['acc_sift_BF'].mean()
    x1 = Results['acc_sift_BF'].max()
    y1 = Results['acc_sift_BF'].min()
    z1 = Results['acc_sift_BF'].std()
    
    m2 = Results['time_sift_BF'].mean()
    x2 = Results['time_sift_BF'].max()
    y2 = Results['time_sift_BF'].min()
    z2 = Results['time_sift_BF'].std()


    mysiftanalysis = mysiftanalysis.append ({'kp':kp, 'Amean':m1,'Amax':x1,'Amin':y1,'Astd':z1,'Tmean':m2,'Tmax':x2,'Tmin':y2,'Tstd':z2},  ignore_index=True )
    
mysiftanalysis.to_csv( 'data/' + TESTNAME + '_siftanalysis'+'.csv')
print ("Data Collection Completed ")  


   
    # ---------- SAVED

#####################################################################################
#############################       VENN DIAGRAM      ###############################
#####################################################################################
 # reading the pickle tree
infile = open('data/Data519_ORIGINAL_Results_mix100_ALL.pickle','rb')
myanalysistestsift = pickle.load(infile)
infile.close()
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted
from matplotlib import pyplot as plt

#################################################
#Missing candidates using accuracy less than 100%
x = list(myanalysistestsift[myanalysistestsift['acc_hsv']<100]['file'])
y = list(myanalysistestsift[myanalysistestsift['acc_rgb']<100]['file'])
z = list(myanalysistestsift[myanalysistestsift['acc_sift_BF']<100]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('missing candidates from accuracy')

###########################################################
#### captured candidates using accuracy greater than 66%
x = list(myanalysistestsift[myanalysistestsift['acc_hsv']>66]['file'])
y = list(myanalysistestsift[myanalysistestsift['acc_rgb']>66]['file'])
z = list(myanalysistestsift[myanalysistestsift['acc_sift_BF']>66]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('captured candidates from accuracy')

##########################################################
#Missing candidates using count less than 4
x = list(myanalysistestsift[myanalysistestsift['Count_hsv']<4]['file'])
y = list(myanalysistestsift[myanalysistestsift['Count_rgb']<4]['file'])
z = list(myanalysistestsift[myanalysistestsift['Count_sift_BF']<4]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('missing candidates from count')


############################################################
#### captured candidates using count greater than 2
x = list(myanalysistestsift[myanalysistestsift['Count_hsv']>2]['file'])
y = list(myanalysistestsift[myanalysistestsift['Count_rgb']>2]['file'])
z = list(myanalysistestsift[myanalysistestsift['Count_sift_BF']>2]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('captured candidates from count')



