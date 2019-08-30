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

############################################################################
#############################  VENN DIAGRAM  ###############################
############################################################################

 # reading the pickle tree
infile = open('data/Data519_original20_Results_mix100_ALL.pickle','rb')
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
plt.title('candidates_missed_Accuracy_20 < 100')

###########################################################
#### captured candidates using accuracy greater than 66%
x = list(myanalysistestsift[myanalysistestsift['acc_hsv']==100]['file'])
y = list(myanalysistestsift[myanalysistestsift['acc_rgb']==100]['file'])
z = list(myanalysistestsift[myanalysistestsift['acc_sift_BF']==100]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('candidates_captured_Accuracy_20 = 100')

##########################################################
#Missing candidates using count less than 4
x = list(myanalysistestsift[myanalysistestsift['Count_hsv']<4]['file'])
y = list(myanalysistestsift[myanalysistestsift['Count_rgb']<4]['file'])
z = list(myanalysistestsift[myanalysistestsift['Count_sift_BF']<4]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('candidates_missed_Count_100 < 4')


############################################################
#### captured candidates using count equal to 4
x = list(myanalysistestsift[myanalysistestsift['Count_hsv']==4]['file'])
y = list(myanalysistestsift[myanalysistestsift['Count_rgb']==4]['file'])
z = list(myanalysistestsift[myanalysistestsift['Count_sift_BF']==4]['file'])
venn3_unweighted([set(x), set(y), set(z)], set_labels = ('hsv', 'rgb', 'sift'))
plt.title('candidates_captured_Count_100 = 4')





##################################################################################
##########  PLOT THRESHOLDING CURVE ################################
##################################################################################

#### check Gaussian nature of scores and success positions
import ImageSearch_Algo_HSV
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import Accuracy as accuracy
from kneed import KneeLocator
import random 
import matplotlib.pyplot as plt

#to reload module: uncomment use the following 
%load_ext autoreload
%autoreload 2


# Load Features 
TESTNAME = 'Data519_original20'
file_HSV_Feature = 'data/' + TESTNAME + '_PandasDF_HSV_Features'
file_HSV_Tree = 'data/' + TESTNAME + '_HSV_Tree'
file_RGB_Feature = 'data/' + TESTNAME + '_PandasDF_RGB_Features'
file_RGB_Tree = 'data/' + TESTNAME + '_RGB_Tree'
file_SIFT_Feature = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp100'
mydataRGB = ImageSearch_Algo_RGB.RGB_LOAD_FEATURES (file_RGB_Feature)
mydataHSV = ImageSearch_Algo_HSV.HSV_LOAD_FEATURES (file_HSV_Feature)
mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature)
# Tree & Clusters 
myRGBtree = ImageSearch_Algo_RGB.RGB_Load_Tree (file_RGB_Tree)
myHSVtree = ImageSearch_Algo_HSV.HSV_Load_Tree (file_HSV_Tree)


# Threshold chart Score vs Sorted Samples
def plot_match_scores(imagematches,label, title): 
    score = []
    successScore = []
    # score, file = item
    for item in imagematches:
        x, y = item
        score.append(x)
    # print(score)
    successPositions =i_rgb
    for i in i_rgb: 
        successScore.append(score[i])
    print ('Success positions', successPositions)

    #  can throw exceptions in case of less points

    knee = 6
    try : 
        elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
        print ('Detected Elbow cluster value :', elbow.knee)
        knee = elbow.knee
    except: 
        pass    
    qualifiedItems = min (knee, 6)

    # plt.scatter ( [counter]*len(imagematches), score, c=matchesposition)
    plt.plot(score)
    plt.scatter(successPositions, successScore, c='r' )
    plt.vlines( qualifiedItems , 0, max(score), colors='g')
    plt.xlabel('n_samples')
    plt.ylabel(label + ' Score')
    plt.title(title)
    plt.savefig('./data/plots/' + title +'.png')
    plt.show()
    




# q_path = random.sample(q_paths, 1)[0]
# q_path = './imagesbooks/ukbench03618.jpg'
# q_path = './imagesbooks/ukbench02720.jpg'
# q_path = './imagesbooks/ukbench06532.jpg'
# q_path = './imagesbooks/ukbench06533.jpg'
# q_path = './imagesbooks/ukbench02718.jpg'
# q_path = './imagesbooks/ukbench00487.jpg'
# q_path = './imagesbooks/ukbench02730.jpg'
# q_path = './imagesbooks/ukbench05941.jpg'
# q_path = './imagesbooks/ukbench10165.jpg'
# q_path = './imagesbooks/ukbench10166.jpg'
# q_path = './imagesbooks/ukbench08595.jpg'
# q_path = './imagesbooks/ukbench08594.jpg'
q_path = './imagesbooks/ukbench05952.jpg'


imagematcheshsv , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, 100)
a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
print (q_path, 'search time', searchtime)
print ('Accuracy =',  a, '%', '| Quality:', d )
print ('Count', cnt, ' | position', i_rgb)
plot_match_scores(imagematcheshsv, 'HSV', '13_HSV_threshold')
# imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSV, q_path)


imagematchesrgb , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, 100)
a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
print (q_path, 'search time', searchtime)
print ('Accuracy =',  a, '%', '| Quality:', d )
print ('Count', cnt, ' | position', i_rgb)
plot_match_scores(imagematchesrgb,'RGB','13_RGB_threshold')
# imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, 0.5)


# imagematchesSIFT , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydataSIFT, q_path, 100, 0.7, 500)
imagematchesSIFT , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF_DIST(mydataSIFT, q_path, 100, 0.7, 100)
a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematchesSIFT, 20)
print (q_path, 'search time', searchtime)
print ('Accuracy =',  a, '%', '| Quality:', d )
print ('Count', cnt, ' | position', i_rgb)
plot_match_scores(imagematchesSIFT, 'SIFT','13_SIFT_threshold')


# plot_match_scores(imagematchesrgb)
# plot_match_scores(imagematcheshsv)
# plot_match_scores(imagematchesSIFT)


# q_path = './imagesbooks/ukbench10165.jpg'
# q_path = './imagesbooks/ukbench10166.jpg'
# q_path = './imagesbooks/ukbench08595.jpg'
# q_path = './imagesbooks/ukbench08594.jpg'
q_path = './imagesbooks/ukbench05952.jpg'



##################################################################################
##########  PLOT DISTRIBUTION 2D SCATTER and 3D  
##################################################################################
import pandas as pd 
# Consolidate all to 1 final table of scores for all images in datset 

hsv_match_score=[]
hsv_match_file=[]
rgb_match_score=[]
rgb_match_file=[]
sift_match_score=[]
sift_match_file=[]

space = len(mydataHSV.index)
# space = 100

imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, space)
for myitem in imagematcheshsv:
    x, y = myitem
    hsv_match_score.append(x)
    hsv_match_file.append(y)
data = { 'file' : hsv_match_file , 'HSVScore' : hsv_match_score }
hsvTable = pd.DataFrame ( data )

imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, space)
for myitem in imagematchesrgb:
    x, y = myitem
    rgb_match_score.append(x)
    rgb_match_file.append(y)
data = { 'file' : rgb_match_file , 'RGBScore' : rgb_match_score }
rgbTable = pd.DataFrame ( data )

imagematchesSIFT, searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF_DIST(mydataSIFT, q_path, 100, 0.75, space)
for myitem in imagematchesSIFT:
    x, y = myitem
    sift_match_score.append(x)
    sift_match_file.append(y)
data = { 'file' : sift_match_file , 'SIFTScore' : sift_match_score }
siftTable = pd.DataFrame ( data )


finalTable = pd.merge(hsvTable,rgbTable, on='file')
finalTable = pd.merge(finalTable,siftTable, on='file')

truth = accuracy.accuracy_groundtruth(q_path) 
finalTable ['Truth'] = 0 
finalTable.loc [finalTable['file'].isin(truth) , 'Truth' ] = 1     
finalTable = finalTable[finalTable.file != q_path]

finalTable = finalTable.sort_values(by=['Truth'])

# plot_match_scores(imagematcheshsv,'HSV', '13_HSV_threshold')
# plot_match_scores(imagematchesrgb,'RGB', '13_RGB_threshold')
# plot_match_scores(imagematchesSIFT,'SIFT', '13_SIFT_threshold')





#################################################################################
##########              PLOT SCATTERS for SCORES            #####################
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

# colormaps 
customcmap = colors.ListedColormap(['green', 'red'])

# Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=40, azim=134)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(list(finalTable['HSVScore']), list(finalTable['RGBScore']), list(finalTable['SIFTScore']), c=list(finalTable['Truth']), s=50 ,cmap=customcmap) 
# marker='o', cmap=plt.cm.RdYlGn)
ax.set_xlabel('HSVScore')
ax.set_ylabel('RGBScore')
ax.set_zlabel('SIFTScore')
plt.title('Score Comparison')
# plt.legend(loc=2)
plt.savefig('./data/plots/' + '13_ScoreComparison' +'.png')
plt.show()


# X, Y Scatter HSV vs RGB 
customcmap = colors.ListedColormap(['green', 'red'])
# customcmap = colors.ListedColormap(['gray', 'red'])
# customcmap = colors.ListedColormap(['cyan', 'magenta'])
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter( list(finalTable['HSVScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=30, cmap=customcmap)
ax.set_title('HSV vs RGB ')
ax.set_xlabel('HSV ')
ax.set_ylabel('RGB ')
plt.savefig('./data/plots/' + '13_HSVvsRGB ' +'.png')
plt.colorbar(scatter)


# X, Y Scatter SIFT vs RGB 
customcmap = colors.ListedColormap(['gray', 'red'])
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter( list(finalTable['SIFTScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=30, cmap=customcmap)
ax.set_title('SIFT vs RGB ')
ax.set_xlabel('SIFT ')
ax.set_ylabel('RGB ')
plt.savefig('./data/plots/' + '13_SIFTvsRGB ' +'.png')
plt.colorbar(scatter)

# # Threshold chart Score vs Sorted Samples
# def plot_match_scores(imagematches): 
#     score = []
#     successScore = []
#     # score, file = item
#     for item in imagematches:
#         x, y = item
#         score.append(x)
#     # print(score)
#     successPositions =i_rgb
#     for i in i_rgb: 
#         successScore.append(score[i])

#     #  can throw exceptions in case of less points

#     knee = 6
#     try : 
#         elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
#         print ('Detected Elbow cluster value :', elbow.knee)
#         knee = elbow.knee
#     except: 
#         pass    
#     qualifiedItems = min (knee, 6)

#     # plt.scatter ( [counter]*len(imagematches), score, c=matchesposition)
#     plt.plot(score)
#     plt.scatter(successPositions, successScore, c='r' )
#     plt.vlines( qualifiedItems , 0, max(score), colors='g')
#     plt.xlabel('n_samples')
#     plt.ylabel('Score')
#     plt.show()











#################################################################
########## Unique problem samples ###################3

myFavList = ['./imagesbooks/ukbench10165.jpg','./imagesbooks/ukbench10166.jpg','./imagesbooks/ukbench08595.jpg','./imagesbooks/ukbench08594.jpg','./imagesbooks/ukbench05952.jpg']



##################################################################################
##########  PLOT DISTRIBUTION 2D SCATTER and 3D  
##################################################################################
import pandas as pd 
import os 
# Consolidate all to 1 final table of scores for all images in datset 

for q_path in myFavList: 
    hsv_match_score=[]
    hsv_match_file=[]
    rgb_match_score=[]
    rgb_match_file=[]
    sift_match_score=[]
    sift_match_file=[]

    space = len(mydataHSV.index)
    # space = 100

    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, space)
    for myitem in imagematcheshsv:
        x, y = myitem
        hsv_match_score.append(x)
        hsv_match_file.append(y)
    data = { 'file' : hsv_match_file , 'HSVScore' : hsv_match_score }
    hsvTable = pd.DataFrame ( data )

    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, space)
    for myitem in imagematchesrgb:
        x, y = myitem
        rgb_match_score.append(x)
        rgb_match_file.append(y)
    data = { 'file' : rgb_match_file , 'RGBScore' : rgb_match_score }
    rgbTable = pd.DataFrame ( data )

    imagematchesSIFT, searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF_DIST(mydataSIFT, q_path, 100, 0.75, space)
    for myitem in imagematchesSIFT:
        x, y = myitem
        sift_match_score.append(x)
        sift_match_file.append(y)
    data = { 'file' : sift_match_file , 'SIFTScore' : sift_match_score }
    siftTable = pd.DataFrame ( data )


    finalTable = pd.merge(hsvTable,rgbTable, on='file')
    finalTable = pd.merge(finalTable,siftTable, on='file')

    truth = accuracy.accuracy_groundtruth(q_path) 
    finalTable ['Truth'] = 0 
    finalTable.loc [finalTable['file'].isin(truth) , 'Truth' ] = 1     
    finalTable = finalTable[finalTable.file != q_path]

    finalTable = finalTable.sort_values(by=['Truth'])

    # plot_match_scores(imagematcheshsv,'HSV', '13_HSV_threshold')
    # plot_match_scores(imagematchesrgb,'RGB', '13_RGB_threshold')
    # plot_match_scores(imagematchesSIFT,'SIFT', '13_SIFT_threshold')





    #################################################################################
    ##########              PLOT SCATTERS for SCORES            #####################
    #################################################################################

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d import Axes3D


    myID = os.path.basename(q_path)
    # colormaps 
    customcmap = colors.ListedColormap(['green', 'red'])

    # Plotting
    fig = plt.figure(1, figsize=(7,7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=40, azim=134)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(list(finalTable['HSVScore']), list(finalTable['RGBScore']), list(finalTable['SIFTScore']), c=list(finalTable['Truth']), s=50 ,cmap=customcmap) 
    # marker='o', cmap=plt.cm.RdYlGn)
    ax.set_xlabel('HSVScore')
    ax.set_ylabel('RGBScore')
    ax.set_zlabel('SIFTScore')
    plt.title(myID + ' Score Comparison')
    # plt.legend(loc=2)
    plt.savefig('./data/plots/' + myID + '_ScoreComparison' +'.png')
    plt.show()


    # X, Y Scatter HSV vs RGB 
    customcmap = colors.ListedColormap(['green', 'red'])
    # customcmap = colors.ListedColormap(['gray', 'red'])
    # customcmap = colors.ListedColormap(['cyan', 'magenta'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter( list(finalTable['HSVScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=30, cmap=customcmap)
    ax.set_title(myID + ' HSV vs RGB ')
    ax.set_xlabel('HSV ')
    ax.set_ylabel('RGB ')
    plt.savefig('./data/plots/' + myID + '_HSVvsRGB ' +'.png')
    plt.colorbar(scatter)


    # X, Y Scatter SIFT vs RGB 
    customcmap = colors.ListedColormap(['gray', 'red'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter( list(finalTable['SIFTScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=30, cmap=customcmap)
    ax.set_title(myID + ' SIFT vs RGB ')
    ax.set_xlabel('SIFT ')
    ax.set_ylabel('RGB ')
    plt.savefig('./data/plots/' + myID + '_SIFTvsRGB ' +'.png')
    plt.colorbar(scatter)




#############################################################################
#######################------  RGB THRESHOLDING  Plot--- ########################
##############################################################################


import ImageSearch_Algo_RGB
from imutils import paths
from matplotlib import colors
import matplotlib.pyplot as plt
import Accuracy as accuracy
import random
from kneed import KneeLocator

# # to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = qpaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydata, mytime = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print ('RGB Generation Time' , mytime)



#------SEARCH TEST------------------------------#

import Accuracy as accuracy

# q_path = random.sample(imagepaths, 1)[0]
q_paths = random.sample(imagepaths, 30)  # random sample 100 items in list
customcmap = colors.ListedColormap(['green', 'red'])

counter = 1 
clist = [] 
qlist = []
for q_path in q_paths: 
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, 100)

    a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
    score = []
    matchesposition = [0]*len(imagematchesrgb)
    for i in i_rgb: 
        matchesposition[i] = 1

    # score, file = item
    for item in imagematchesrgb:
        x, y = item
        score.append(x)
    # print(score)

    knee = 6
    try : 
        elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
        # print ('Detected Elbow cluster value :', elbow.knee)
        knee = elbow.knee
    except: 
        pass    
    qualifiedItems = min (knee, 6)

    plt.scatter ( [counter]*len(imagematchesrgb), score, c=matchesposition, cmap=customcmap, s=10)

    clist.append(counter)
    qlist.append(score[qualifiedItems]-0.25) # fixed offset 
    counter +=1 # DONT FORGET 

    # import matplotlib.pyplot as plt 
    # plt.scatter ( [counter]*len(imagematchesrgb), score, c=matchesposition)

# plot line chart 
plt.step ( clist, qlist, where='mid', c='purple')   
# plt.plot ( clist, qlist, '-')   


plt.colorbar()
plt.xlabel('n_samples')
plt.ylabel('RGB Score')
plt.title('RGB_Score_based_Thresholding')
plt.savefig('./data/plots/' + 'RGB_Thresholding' +'.png')
plt.show()


    

##############################################################################
#######################----HSV THRESHOLDING Plot------ ########################
##############################################################################


import matplotlib.pyplot as plt
from matplotlib import colors
import Accuracy as accuracy
import random
import ImageSearch_Algo_HSV
from kneed import knee_locator

# # to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytime)

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')


#------SEARCH TEST------------------------------#

q_paths = random.sample(imagepaths, 30)  # random sample 100 items in list
customcmap = colors.ListedColormap(['green', 'red'])
counter = 1 
clist = [] 
qlist = []

for q_path in q_paths: 
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, 100)
       
    a, d, i_hsv, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
    score = []
    matchesposition = [0]*len(imagematcheshsv)
    for i in i_hsv: 
        matchesposition[i] = 1

    # score, file = item
    for item in imagematcheshsv:
        x, y = item
        score.append(x)
    # print (score)

    knee = 6
    try : 
        elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
        # print ('Detected Elbow cluster value :', elbow.knee)
        knee = elbow.knee
    except: 
        pass    
    qualifiedItems = min (knee, 6)

    plt.scatter ( [counter]*len(imagematcheshsv), score, c=matchesposition, cmap=customcmap, s=10)

    clist.append(counter)
    qlist.append(score[qualifiedItems]-0.25) # fixed offset 
    counter +=1 # DONT FORGET 

    # import matplotlib.pyplot as plt 
    # plt.scatter ( [counter]*len(imagematchesrgb), score, c=matchesposition)

# plot line chart 
plt.step ( clist, qlist, where='mid', c='purple')   
# plt.plot ( clist, qlist, '-')      

plt.colorbar()
plt.xlabel('n_samples')
plt.ylabel('HSV Score')
plt.title('HSV_Score_based_Thresholding')
plt.savefig('./data/plots/' + 'HSV_Thresholding' +'.png')
plt.show()



#####################################################################
########  STATISTICAL HSV vs. RGB  score/thresholding comparison
#####################################################################

import pandas as pd

# X, Y Scatter HSV vs RGB 
customcmap = colors.ListedColormap(['green', 'red'])
# customcmap = colors.ListedColormap(['gray', 'red'])
# customcmap = colors.ListedColormap(['cyan', 'magenta'])
fig = plt.figure()
ax = fig.add_subplot(111)

counter = 0 
for q_path in q_paths[:30]: 
    
    counter +=1 # DONT FORGET 

    hsv_match_score=[]
    hsv_match_file=[]
    rgb_match_score=[]
    rgb_match_file=[]

    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, 50)
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, 50)
    space = 100
    # create a dataframe for HSV ans RGB 
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, space)
    for myitem in imagematcheshsv:
        x, y = myitem
        hsv_match_score.append(x)
        hsv_match_file.append(y)
    data = { 'file' : hsv_match_file , 'HSVScore' : hsv_match_score }
    hsvTable = pd.DataFrame ( data )

    for myitem in imagematchesrgb:
        x, y = myitem
        rgb_match_score.append(x)
        rgb_match_file.append(y)
    data = { 'file' : rgb_match_file , 'RGBScore' : rgb_match_score }
    rgbTable = pd.DataFrame ( data )

    finalTable = pd.merge(hsvTable,rgbTable, on='file')
    truth = accuracy.accuracy_groundtruth(q_path) 
    finalTable ['Truth'] = 0 
    finalTable.loc [finalTable['file'].isin(truth) , 'Truth' ] = 1     
    finalTable = finalTable[finalTable.file != q_path]
    finalTable = finalTable.sort_values(by=['Truth']) # sort by Truth for plotting overlay

    scatter = ax.scatter( list(finalTable['HSVScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=30, cmap=customcmap)

ax.set_title('HSV vs RGB Thresholding comparison ')
ax.set_xlabel('HSV ')
ax.set_ylabel('RGB ')
plt.colorbar(scatter)
# plt.savefig('./data/plots/' + 'HSV vs RGB_Thresholding' +'.png')
plt.show()
