


#--------------------------SIFT ALGO DATA-------------------------------#
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
import  ImageSearch_Algo_SIFT

IMGDIR = "./imagesbooks/"


# ------------ GENERATION TEST

# Hyper-Parameters for SIFT comparison
sift_features_limit = 1000
lowe_ratio = 0.75
predictions_count = 50

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
mydata1, mytime1 = gen_sift_features(imagepaths, 1000)




# ------------------ SEARCH TEST 


# q_path = random.sample(fileterd_file, 1)[0]
# q_path = './imagesbooks/ukbench00481.jpg'
# imagepredictions , searchtime = SIFT_SEARCH(mydata1, q_path, 50 ,0.75, 20)
imagepredictions , searchtime = SIFT_SEARCH(filtered, q_path, 50 ,0.75, 20)

# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a,d = accuracy.accuracy_matches(q_path, imagepredictions, 50)
print ('Accuracy =',  a, '%', d, searchtime)

import ImageSearch_Plots as myplots
myplots.plot_predictions(imagepredictions[:20], q_path)


#---------------- Compile data and plot results 


# q_paths = random.sample(imagepaths, 50)  # random sample 100 items in list
# q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg']
q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg', './imagesbooks/ukbench00248.jpg', './imagesbooks/ukbench06408.jpg', './imagesbooks/ukbench00303.jpg', './imagesbooks/ukbench03124.jpg', './imagesbooks/ukbench05776.jpg', './imagesbooks/ukbench06113.jpg', './imagesbooks/ukbench05964.jpg', './imagesbooks/ukbench10164.jpg', './imagesbooks/ukbench02750.jpg', './imagesbooks/ukbench05951.jpg', './imagesbooks/ukbench05983.jpg', './imagesbooks/ukbench03867.jpg', './imagesbooks/ukbench05883.jpg', './imagesbooks/ukbench06049.jpg', './imagesbooks/ukbench06017.jpg', './imagesbooks/ukbench06150.jpg', './imagesbooks/ukbench06151.jpg', './imagesbooks/ukbench02749.jpg', './imagesbooks/ukbench02721.jpg', './imagesbooks/ukbench05879.jpg', './imagesbooks/ukbench06148.jpg', './imagesbooks/ukbench05880.jpg', './imagesbooks/ukbench05929.jpg', './imagesbooks/ukbench06048.jpg', './imagesbooks/ukbench08544.jpg', './imagesbooks/ukbench03058.jpg', './imagesbooks/ukbench10154.jpg', './imagesbooks/ukbench00000.jpg', './imagesbooks/ukbench05972.jpg', './imagesbooks/ukbench05872.jpg', './imagesbooks/ukbench08542.jpg', './imagesbooks/ukbench06004.jpg', './imagesbooks/ukbench05993.jpg', './imagesbooks/ukbench05988.jpg', './imagesbooks/ukbench00483.jpg', './imagesbooks/ukbench08546.jpg', './imagesbooks/ukbench06539.jpg', './imagesbooks/ukbench02748.jpg', './imagesbooks/ukbench05980.jpg', './imagesbooks/ukbench08001.jpg', './imagesbooks/ukbench03890.jpg', './imagesbooks/ukbench03059.jpg', './imagesbooks/ukbench10081.jpg', './imagesbooks/ukbench06519.jpg', './imagesbooks/ukbench05787.jpg']

import Accuracy as accuracy  
keypoints = [50 ]
# 100 ,300, 500, 700, 900
accStats = pd.DataFrame(columns=['file', 'PCount'])
for q_path in q_paths: 
    row_dict = {'file' : q_path }
    
    for item in keypoints:  
        print ("Processing, time", q_paths.index(q_path), searchtime, item)
        imagepredictions , searchtime = SIFT_SEARCH(mydata1, q_path, item, 0.75, 50)
        a10,d, i = accuracy.accuracy_matches(q_path, imagepredictions, 10 )
        # a20,d = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
        # a30,d = accuracy.accuracy_matches(q_path, imagepredictions, 30 )
        # a50,d = accuracy.accuracy_matches(q_path, imagepredictions, 50 )
        row_dict['kp_' + str(item) + '_predict10'] = a10
        # row_dict['kp_' + str(item)+ '_predict20'] = a20
        # row_dict['kp_' + str(item)+ '_predict30'] = a30
        # row_dict['kp_' + str(item) +'_predict50'] = a50
        row_dict['kp_'+str(item)+'_quality'] = d
        row_dict['kp_'+str(item)+'_time'] = searchtime
        row_dict['kp_'+str(item)+'matchposition'] = i
        

    accStats = accStats.append( row_dict , ignore_index=True)

# plt.plot(accStats)
# plt.plot (accStats['PCount'])
plt.hlines(accStats.mean(), 0, 100, 'r')
print ("Mean Acc = ", accStats.mean())

accStats.to_csv('accStatsSiftmatchposition2.csv')



#changing training lib size----------------------------------------#
# library size vs searchtime study
sortedmydata1 = mydata1.sort_values('file')

filtered = sortedmydata1.head(500)
# filtered100 = sortedmydata1.head(100)
# filtered200 = sortedmydata1.head(200)
# filtered300 = sortedmydata1.head(300)
# filtered400 = sortedmydata1.head(400)
# filtered500 = sortedmydata1.head(500)

# filtered = [filtered50, filtered100, filtered200, filtered300, filtered400, filtered500]
# print (filtered)

accStats1 = pd.DataFrame(columns=['file', 'Acc', 'Stime'])


fileterd_file = list(filtered['file'])
# print( fileterd_file)
q_paths = random.sample(fileterd_file, 50)  # random sample 100 items in list 

for q_path in q_paths: 
       
    print ("Processing, time", q_paths.index(q_path), searchtime)
    imagepredictions , searchtime = SIFT_SEARCH(filtered, q_path, 50,0.75, 20)
    a = accuracy.accuracy_matches(q_path, imagepredictions, 20)
    accStats1 = accStats1.append({ 'file': q_path, 'Acc': a, 'Stime': searchtime } , ignore_index=True)
    


# plt.plot(accStats1['Acc'])
# plt.hlines(accStats1['Acc'].mean(), 0, 100, 'r')
print ('Accuracy =',  a, '%', searchtime)
print ("Mean Acc = ", accStats1['Acc'].mean())
accStats1.to_csv('accStats1Siftlib500.csv')








#--------------------RGB ALGO DATA --------------------------------#

# ------------- GENERATION TEST-------------------#

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydata, mytime = RGB_GEN(imagepaths)




#------SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]
imagematches , searchtime = RGB_SEARCH(mydata, q_path, 0.7)

# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a = accuracy.accuracy_matches(q_path, imagematches, 50)
print ('Accuracy =',  a, '%')

import ImageSearch_Plots as myplots
myplots.plot_predictions(imagematches, q_path)


#---------------- Compile data and plot results 

accStats = pd.DataFrame(columns=['file','Acc', 'PCount'])

q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 

for q_path in q_paths:    
    imagematches , searchtime = RGB_SEARCH(mydata, q_path, 0.7)
    a = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': len(imagematches) } , ignore_index=True)


plt.plot(accStats['Acc'])
plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print ("Mean Acc = ", accStats['Acc'].mean())