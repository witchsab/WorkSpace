


#################################################################################
############################     MERGE CODE  + PLOT   ##########################
#################################################################################



import PIL
from PIL import Image
import imagehash
import os
import cv2
import time
from pprint import pprint
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd 
import random
import ImageSearch_Algo_HSV 
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import Accuracy as accuracy  
from kneed import KneeLocator

# # to reload module: uncomment use the following
%load_ext autoreload
%autoreload 2

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytimeHSV = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytimeHSV)

mydataRGB, mytimeRGB = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print ('RGB Feature Generation time', mytimeRGB)

#Loading kp100 sift features
mydatasift = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES('data/SIFT_features_519set_kp100_pandas')


# to create a new tree from dataframe features 'mydataHSV'
mytreeHSV = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')
mytreeRGB = ImageSearch_Algo_RGB.RGB_Create_Tree (mydataRGB, savefile='RGB_Tree')

# q_paths = random.sample(imagepaths, 5)  # random sample 100 items in list

# q_paths = imagepaths  

# q_paths = ['./imagesbooks/ukbench05960.jpg']  #,'./imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg']
# q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg', './imagesbooks/ukbench00248.jpg', './imagesbooks/ukbench06408.jpg', './imagesbooks/ukbench00303.jpg', './imagesbooks/ukbench03124.jpg', './imagesbooks/ukbench05776.jpg', './imagesbooks/ukbench06113.jpg', './imagesbooks/ukbench05964.jpg', './imagesbooks/ukbench10164.jpg', './imagesbooks/ukbench02750.jpg', './imagesbooks/ukbench05951.jpg', './imagesbooks/ukbench05983.jpg', './imagesbooks/ukbench03867.jpg', './imagesbooks/ukbench05883.jpg', './imagesbooks/ukbench06049.jpg', './imagesbooks/ukbench06017.jpg', './imagesbooks/ukbench06150.jpg', './imagesbooks/ukbench06151.jpg', './imagesbooks/ukbench02749.jpg', './imagesbooks/ukbench02721.jpg', './imagesbooks/ukbench05879.jpg', './imagesbooks/ukbench06148.jpg', './imagesbooks/ukbench05880.jpg', './imagesbooks/ukbench05929.jpg', './imagesbooks/ukbench06048.jpg', './imagesbooks/ukbench08544.jpg', './imagesbooks/ukbench03058.jpg', './imagesbooks/ukbench10154.jpg', './imagesbooks/ukbench00000.jpg', './imagesbooks/ukbench05972.jpg', './imagesbooks/ukbench05872.jpg', './imagesbooks/ukbench08542.jpg', './imagesbooks/ukbench06004.jpg', './imagesbooks/ukbench05993.jpg', './imagesbooks/ukbench05988.jpg', './imagesbooks/ukbench00483.jpg', './imagesbooks/ukbench08546.jpg', './imagesbooks/ukbench06539.jpg', './imagesbooks/ukbench02748.jpg', './imagesbooks/ukbench05980.jpg', './imagesbooks/ukbench08001.jpg', './imagesbooks/ukbench03890.jpg', './imagesbooks/ukbench03059.jpg', './imagesbooks/ukbench10081.jpg', './imagesbooks/ukbench06519.jpg', './imagesbooks/ukbench05787.jpg']

q_paths = ["./imagesbooks/ukbench00196.jpg",  "./imagesbooks/ukbench00199.jpg",  "./imagesbooks/ukbench00296.jpg",  "./imagesbooks/ukbench00298.jpg",  "./imagesbooks/ukbench00299.jpg",  "./imagesbooks/ukbench00300.jpg",  "./imagesbooks/ukbench00302.jpg",  "./imagesbooks/ukbench00303.jpg",  "./imagesbooks/ukbench02730.jpg",  "./imagesbooks/ukbench02740.jpg",  "./imagesbooks/ukbench02743.jpg",  "./imagesbooks/ukbench05608.jpg",  "./imagesbooks/ukbench05932.jpg",  "./imagesbooks/ukbench05933.jpg",  "./imagesbooks/ukbench05934.jpg",  "./imagesbooks/ukbench05935.jpg",  "./imagesbooks/ukbench05952.jpg",  "./imagesbooks/ukbench05953.jpg",  "./imagesbooks/ukbench05954.jpg",  "./imagesbooks/ukbench05955.jpg",  "./imagesbooks/ukbench05956.jpg",  "./imagesbooks/ukbench05957.jpg",  "./imagesbooks/ukbench05958.jpg",  "./imagesbooks/ukbench05959.jpg",  "./imagesbooks/ukbench06148.jpg",  "./imagesbooks/ukbench06149.jpg",  "./imagesbooks/ukbench06150.jpg",  "./imagesbooks/ukbench06151.jpg",  "./imagesbooks/ukbench06558.jpg",  "./imagesbooks/ukbench06559.jpg",  "./imagesbooks/ukbench07285.jpg",  "./imagesbooks/ukbench07588.jpg",  "./imagesbooks/ukbench07589.jpg",  "./imagesbooks/ukbench07590.jpg",  "./imagesbooks/ukbench08540.jpg",  "./imagesbooks/ukbench08542.jpg",  "./imagesbooks/ukbench08592.jpg", "./imagesbooks/ukbench08594.jpg","./imagesbooks/ukbench08595.jpg","./imagesbooks/ukbench08609.jpg","./imagesbooks/ukbench09364.jpg","./imagesbooks/ukbench09365.jpg","./imagesbooks/ukbench09366.jpg","./imagesbooks/ukbench10061.jpg","./imagesbooks/ukbench10065.jpg","./imagesbooks/ukbench10066.jpg","./imagesbooks/ukbench10085.jpg","./imagesbooks/ukbench10087.jpg","./imagesbooks/ukbench10108.jpg","./imagesbooks/ukbench10109.jpg","./imagesbooks/ukbench10110.jpg","./imagesbooks/ukbench10112.jpg","./imagesbooks/ukbench10113.jpg","./imagesbooks/ukbench10114.jpg","./imagesbooks/ukbench10116.jpg","./imagesbooks/ukbench10118.jpg","./imagesbooks/ukbench10119.jpg","./imagesbooks/ukbench10124.jpg","./imagesbooks/ukbench10125.jpg","./imagesbooks/ukbench10126.jpg","./imagesbooks/ukbench10128.jpg","./imagesbooks/ukbench10129.jpg","./imagesbooks/ukbench10130.jpg","./imagesbooks/ukbench10131.jpg","./imagesbooks/ukbench10152.jpg","./imagesbooks/ukbench10153.jpg","./imagesbooks/ukbench10154.jpg","./imagesbooks/ukbench10164.jpg","./imagesbooks/ukbench10165.jpg","./imagesbooks/ukbench10166.jpg","./imagesbooks/ukbench10167.jpg"]

# q_path = "./imagesbooks/ukbench05959.jpg"
q_path = random.sample(q_paths, 1)[0]  # random sample 100 items in list
print (q_path)

hsv_match_score=[]
hsv_match_file=[]
rgb_match_score=[]
rgb_match_file=[]
sift_match_score=[]
sift_match_file=[]



imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (mytreeHSV, mydataHSV, q_path, len(mydataHSV.index))
for myitem in imagematcheshsv:
    x, y = myitem
    hsv_match_score.append(x)
    hsv_match_file.append(y)
data = { 'file' : hsv_match_file , 'HSVScore' : hsv_match_score }
hsvTable = pd.DataFrame ( data )



imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (mytreeRGB, mydataRGB, q_path, 519)
for myitem in imagematchesrgb:
    x, y = myitem
    rgb_match_score.append(x)
    rgb_match_file.append(y)
data = { 'file' : rgb_match_file , 'RGBScore' : rgb_match_score }
rgbTable = pd.DataFrame ( data )




imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydatasift, q_path, 100, 0.6, 519)
for myitem in imagepredictions:
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


# finalTable.to_csv('finalscoreTable.csv')


# Create plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



# Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=40, azim=134)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(list(finalTable['HSVScore']), list(finalTable['RGBScore']), list(finalTable['SIFTScore']), c=list(finalTable['Truth']), marker='o')
ax.set_xlabel('HSVScore')
ax.set_ylabel('RGBScore')
ax.set_zlabel('SIFTScore')
plt.title('Score Comparison')
# plt.legend(loc=2)
plt.show()


# X, Y Scatter HSV vs RGB 
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter( list(finalTable['HSVScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=10)
ax.set_title('HSV vs RGB ')
ax.set_xlabel('HSV ')
ax.set_ylabel('RGB ')
plt.colorbar(scatter)



# X, Y Scatter SIFT vs RGB 
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter( list(finalTable['SIFTScore']), list(finalTable['RGBScore']), c= list(finalTable["Truth"]),s=10)
ax.set_title('SIFT vs RGB ')
ax.set_xlabel('SIFT ')
ax.set_ylabel('RGB ')
plt.colorbar(scatter)












