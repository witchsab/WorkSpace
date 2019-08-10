


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
imagepredictions , searchtime = ImageSearch_Algo_SIFT ( mydatasift, q_path, 100 ,0.75, 20)

# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a, b,c,d = accuracy.accuracy_matches(q_path, imagepredictions, 50)
print ('Accuracy =',  a, '%', d, searchtime)

# import ImageSearch_Plots as myplots
# myplots.plot_predictions(imagepredictions[:20], q_path)


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

import ImageSearch_Algo_RGB

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydata, mytime = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print (mytime)



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

# without tree
# q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg', './imagesbooks/ukbench00248.jpg', './imagesbooks/ukbench06408.jpg', './imagesbooks/ukbench00303.jpg', './imagesbooks/ukbench03124.jpg', './imagesbooks/ukbench05776.jpg', './imagesbooks/ukbench06113.jpg', './imagesbooks/ukbench05964.jpg', './imagesbooks/ukbench10164.jpg', './imagesbooks/ukbench02750.jpg', './imagesbooks/ukbench05951.jpg', './imagesbooks/ukbench05983.jpg', './imagesbooks/ukbench03867.jpg', './imagesbooks/ukbench05883.jpg', './imagesbooks/ukbench06049.jpg', './imagesbooks/ukbench06017.jpg', './imagesbooks/ukbench06150.jpg', './imagesbooks/ukbench06151.jpg', './imagesbooks/ukbench02749.jpg', './imagesbooks/ukbench02721.jpg', './imagesbooks/ukbench05879.jpg', './imagesbooks/ukbench06148.jpg', './imagesbooks/ukbench05880.jpg', './imagesbooks/ukbench05929.jpg', './imagesbooks/ukbench06048.jpg', './imagesbooks/ukbench08544.jpg', './imagesbooks/ukbench03058.jpg', './imagesbooks/ukbench10154.jpg', './imagesbooks/ukbench00000.jpg', './imagesbooks/ukbench05972.jpg', './imagesbooks/ukbench05872.jpg', './imagesbooks/ukbench08542.jpg', './imagesbooks/ukbench06004.jpg', './imagesbooks/ukbench05993.jpg', './imagesbooks/ukbench05988.jpg', './imagesbooks/ukbench00483.jpg', './imagesbooks/ukbench08546.jpg', './imagesbooks/ukbench06539.jpg', './imagesbooks/ukbench02748.jpg', './imagesbooks/ukbench05980.jpg', './imagesbooks/ukbench08001.jpg', './imagesbooks/ukbench03890.jpg', './imagesbooks/ukbench03059.jpg', './imagesbooks/ukbench10081.jpg', './imagesbooks/ukbench06519.jpg', './imagesbooks/ukbench05787.jpg']


# q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 
accStatsrgb200 = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file':q_path }   
    imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH(mydata, q_path, 0.7)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['rgb' + '_predict10'] = a
    row_dict['rgb'+ '_quality'] = d
    row_dict['rgb'+'_time'] = searchtime
    row_dict['rgb'+'matchposition'] = i
    row_dict['rgb'+'PCount'] = cnt


    accStatsrgb200 = accStatsrgb200.append( row_dict , ignore_index=True)

   


# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStatsrgb['Acc'].mean())
accStatsrgb200.to_csv('accStatsrgb200.csv')





# ----- Alternative tree search code [Optimized search time ]

# test TREE SEARCH code 

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_RGB.RGB_Create_Tree (mydata, savefile='RGB_Tree')

# to load an existing tree 
# thistree = ImageSearch_Algo_RGB.RGB_Load_Tree('RGB_Tree')

accStatsrgb200 = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file':q_path }   
    imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (mytree, mydata, q_path, 100)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['rgb' + '_predict10'] = a
    row_dict['rgb'+ '_quality'] = d
    row_dict['rgb'+'_time'] = searchtime
    row_dict['rgb'+'matchposition'] = i
    row_dict['rgb'+'PCount'] = cnt



    accStatsrgb200 = accStatsrgb200.append( row_dict , ignore_index=True)

   


# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStatsrgb['Acc'].mean())
accStatsrgb200.to_csv('accStatsrgbtree200.csv')


print ('RGB Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)







############---RGB Libsize vs time, scalability----#####

import ImageSearch_Algo_RGB

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataRGB, mytimeRGB = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print ('RGB feature generatiom time ', mytimeRGB)

sortedmydataRGB = mydataRGB.sort_values('file')

mytree = ImageSearch_Algo_RGB.RGB_Create_Tree (sortedmydataRGB, savefile='RGB_Tree')

# to load an existing tree 
# thistree = ImageSearch_Algo_RGB.RGB_Load_Tree('RGB_Tree')

filtered = sortedmydataRGB.head(500)
fileterd_file = list(filtered['file'])
# print( fileterd_file)
q_paths = random.sample(fileterd_file, 50)  # random sample 100 items in list 

accStatsRGBlibsize = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file':q_path }   
    imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (mytree, sortedmydataRGB, q_path, 100)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['rgb' + '_predict10'] = a
    row_dict['rgb'+ '_quality'] = d
    row_dict['rgb'+'_time'] = searchtime
    row_dict['rgb'+'matchposition'] = i
    row_dict['rgb'+'PCount'] = cnt



    accStatsRGBlibsize = accStatsRGBlibsize.append( row_dict , ignore_index=True)

   
# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStatsrgb['Acc'].mean())
accStatsRGBlibsize.to_csv('accStatsRGBlibsize500.csv')


print ('RGB Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)
















###########################---------HSV tree


# -------------HSV RGENERATION TEST-------------------#

import ImageSearch_Algo_HSV 

imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytime)



# test TREE SEARCH code 

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')

# to load an existing tree 
# thistree = ImageSearch_Algo_HSV.HSV_Load_Tree('HSV_Tree')



accStatshsvtree200 = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file':q_path }   
    imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (mytree, mydataHSV, q_path, 100)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['hsv' + '_predict20'] = a
    row_dict['hsv'+ '_quality'] = d
    row_dict['hsv'+'_time'] = searchtime
    row_dict['hsv'+'matchposition'] = i
    row_dict['hsv'+'PCount'] = cnt

    accStatshsvtree200 = accStatshsvtree200.append( row_dict , ignore_index=True)

# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStatshsv['Acc'].mean())
# accStatshsvtree200.to_csv('accStatshsvtree200.csv')


print ('HSV Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)



#### HSV Library size changed---------------
import ImageSearch_Algo_HSV 

imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytime)


sortedmydataHSV = mydataHSV.sort_values('file')

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_HSV.HSV_Create_Tree (sortedmydataHSV, savefile='HSV_Tree')


filtered = sortedmydataHSV.head(100)
fileterd_file = list(filtered['file'])
# print( fileterd_file)
q_paths = random.sample(fileterd_file, 50)  # random sample 100 items in list 

accStatshsvlibsize = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file':q_path }   
    imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (mytree, sortedmydataHSV, q_path, 100)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['hsv100' + '_predict20'] = a
    row_dict['hsv100'+ '_quality'] = d
    row_dict['hsv100'+'_time'] = searchtime
    row_dict['hsv100'+'matchposition'] = i
    row_dict['hsv100'+'PCount'] = cnt

    accStatshsvlibsize = accStatshsvlibsize.append( row_dict , ignore_index=True)

# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStatshsv['Acc'].mean())
accStatshsvlibsize.to_csv('accStatshsvlibsize100.csv')


print ('HSV Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)









#########--------------------------------SIFT


import ImageSearch_Algo_SIFT
#generation
# Hyper-Parameters for SIFT comparison
sift_features_limit = 100
lowe_ratio = 0.75
predictions_count = 50

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
mydatasift, mytimesift = ImageSearch_Algo_SIFT.gen_sift_features(imagepaths, 200)
print('generationtime', mytimesift)


savefile = 'data/SIFT_features_519set_kp200_pandas'
ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydatasift, savefile)
# mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES(savefile)


# q_paths = random.sample(imagepaths, 50)  # random sample 100 items in list

# q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg']
q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg', './imagesbooks/ukbench00248.jpg', './imagesbooks/ukbench06408.jpg', './imagesbooks/ukbench00303.jpg', './imagesbooks/ukbench03124.jpg', './imagesbooks/ukbench05776.jpg', './imagesbooks/ukbench06113.jpg', './imagesbooks/ukbench05964.jpg', './imagesbooks/ukbench10164.jpg', './imagesbooks/ukbench02750.jpg', './imagesbooks/ukbench05951.jpg', './imagesbooks/ukbench05983.jpg', './imagesbooks/ukbench03867.jpg', './imagesbooks/ukbench05883.jpg', './imagesbooks/ukbench06049.jpg', './imagesbooks/ukbench06017.jpg', './imagesbooks/ukbench06150.jpg', './imagesbooks/ukbench06151.jpg', './imagesbooks/ukbench02749.jpg', './imagesbooks/ukbench02721.jpg', './imagesbooks/ukbench05879.jpg', './imagesbooks/ukbench06148.jpg', './imagesbooks/ukbench05880.jpg', './imagesbooks/ukbench05929.jpg', './imagesbooks/ukbench06048.jpg', './imagesbooks/ukbench08544.jpg', './imagesbooks/ukbench03058.jpg', './imagesbooks/ukbench10154.jpg', './imagesbooks/ukbench00000.jpg', './imagesbooks/ukbench05972.jpg', './imagesbooks/ukbench05872.jpg', './imagesbooks/ukbench08542.jpg', './imagesbooks/ukbench06004.jpg', './imagesbooks/ukbench05993.jpg', './imagesbooks/ukbench05988.jpg', './imagesbooks/ukbench00483.jpg', './imagesbooks/ukbench08546.jpg', './imagesbooks/ukbench06539.jpg', './imagesbooks/ukbench02748.jpg', './imagesbooks/ukbench05980.jpg', './imagesbooks/ukbench08001.jpg', './imagesbooks/ukbench03890.jpg', './imagesbooks/ukbench03059.jpg', './imagesbooks/ukbench10081.jpg', './imagesbooks/ukbench06519.jpg', './imagesbooks/ukbench05787.jpg']

# mypaths = random.sample(imagepaths, 200)

# q_paths = list(mypaths)
# # print (q_paths)

import Accuracy as accuracy  
keypoints = [50]
# 100 ,300, 500, 700, 900
accStatssiftkp100 = pd.DataFrame(columns=['file', 'PCount'])
for q_path in q_paths: 
    row_dict = {'file' : q_path }
    
    for item in keypoints:  
        
        imagepredictions , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydatasift, q_path, item, 0.75, 50)
        a ,d, i, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    
        row_dict['kp_' + str(item) + '_predict20'] = a
        # row_dict['kp_' + str(item)+ '_predict20'] = a20
        # row_dict['kp_' + str(item)+ '_predict30'] = a30
        # row_dict['kp_' + str(item) +'_predict50'] = a50
        row_dict['kp_'+str(item)+'_quality'] = d
        row_dict['kp_'+str(item)+'_time'] = searchtime
        row_dict['kp_'+str(item)+'matchposition'] = i
        row_dict['hsv'+'PCount'] = cnt

        print ("Processing, time", q_paths.index(q_path), searchtime, item)

    accStatssiftkp100 = accStatssiftkp100.append( row_dict , ignore_index=True)

# plt.plot(accStats)
# plt.plot (accStats['PCount'])
plt.hlines(accStatssiftkp100.mean(), 0, 100, 'r')
print ("Mean Acc = ", accStatssiftkp100.mean())

accStatssiftkp100.to_csv('accStatsSift_kp_train300_search100.csv')



# sample test [ DO NOT USE ]
q_path = './imagesbooks/ukbench08595.jpg'
q_path = './imagesbooks/ukbench05779.jpg'
q_path = './imagesbooks/ukbench02722.jpg'
imagepredictions , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydatasift, q_path, 100, 0.75, 50)
a ,d, i, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
print (a,d,i,cnt, searchtime)
imagepredictions , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydatasift, q_path, 100, 0.75, 50)
a ,d, i, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
print (a,d,i,cnt, searchtime)





#################################################################################
#############################     MERGE CODE  OLD  ##############################
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

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytimeHSV = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytimeHSV)

mydataRGB, mytimeRGB = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print ('RGB Feature Generation time', mytimeRGB)

mydatasift = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES('data/SIFT_features_519set_kp100_pandas')


# to create a new tree from dataframe features 'mydataHSV'
mytreeHSV = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')
mytreeRGB = ImageSearch_Algo_RGB.RGB_Create_Tree (mydataRGB, savefile='RGB_Tree')

# q_paths = random.sample(imagepaths, 50)  # random sample 100 items in list

# q_paths = ['./imagesbooks/ukbench05960.jpg','./imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg']

# q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg', './imagesbooks/ukbench00248.jpg', './imagesbooks/ukbench06408.jpg', './imagesbooks/ukbench00303.jpg', './imagesbooks/ukbench03124.jpg', './imagesbooks/ukbench05776.jpg', './imagesbooks/ukbench06113.jpg', './imagesbooks/ukbench05964.jpg', './imagesbooks/ukbench10164.jpg', './imagesbooks/ukbench02750.jpg', './imagesbooks/ukbench05951.jpg', './imagesbooks/ukbench05983.jpg', './imagesbooks/ukbench03867.jpg', './imagesbooks/ukbench05883.jpg', './imagesbooks/ukbench06049.jpg', './imagesbooks/ukbench06017.jpg', './imagesbooks/ukbench06150.jpg', './imagesbooks/ukbench06151.jpg', './imagesbooks/ukbench02749.jpg', './imagesbooks/ukbench02721.jpg', './imagesbooks/ukbench05879.jpg', './imagesbooks/ukbench06148.jpg', './imagesbooks/ukbench05880.jpg', './imagesbooks/ukbench05929.jpg', './imagesbooks/ukbench06048.jpg', './imagesbooks/ukbench08544.jpg', './imagesbooks/ukbench03058.jpg', './imagesbooks/ukbench10154.jpg', './imagesbooks/ukbench00000.jpg', './imagesbooks/ukbench05972.jpg', './imagesbooks/ukbench05872.jpg', './imagesbooks/ukbench08542.jpg', './imagesbooks/ukbench06004.jpg', './imagesbooks/ukbench05993.jpg', './imagesbooks/ukbench05988.jpg', './imagesbooks/ukbench00483.jpg', './imagesbooks/ukbench08546.jpg', './imagesbooks/ukbench06539.jpg', './imagesbooks/ukbench02748.jpg', './imagesbooks/ukbench05980.jpg', './imagesbooks/ukbench08001.jpg', './imagesbooks/ukbench03890.jpg', './imagesbooks/ukbench03059.jpg', './imagesbooks/ukbench10081.jpg', './imagesbooks/ukbench06519.jpg', './imagesbooks/ukbench05787.jpg']


accStatsmerge = pd.DataFrame(columns=['file'])
matcheshsv = []
matchesrgb = []

for q_path in q_paths: 

    start = time.time()

    row_dict = {'file':q_path }   
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (mytreeHSV, mydataHSV, q_path, 100)
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (mytreeRGB, mydataRGB, q_path, 100)
    matcheshsv.append((q_path, imagematcheshsv))
    matchesrgb.append((q_path, imagematchesrgb))
    a, d, i_hsv, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
    row_dict['hsv_acc'] = a
    row_dict['hsv_matchindex'] = i_hsv
    row_dict['hsvCount'] = cnt

    a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
    row_dict['rgb_acc'] = a
    row_dict['rgb_matchindex'] = i_rgb
    row_dict['rgbCount'] = cnt

    # create a mergelist 
    mergelist = set ()

    # searchFile, searchResults = matcheshsv[0]

    for myitem in imagematcheshsv:
        x, y = myitem
        mergelist.add(y)

    # searchFile, searchResults = matchesrgb[0]
    for myitem in imagematchesrgb:
        x, y = myitem
        mergelist.add(y)

    mergelist = list(mergelist)

    print ("Candidate count ", len(mergelist))
    row_dict['mergecount'] = len(mergelist)
    

    filteredsift = mydatasift[mydatasift['file'].isin(mergelist)]

    # df = pd.DataFrame({'A' : [5,6,3,4], 'B' : [1,2,3,5]})
    # df[df['A'].isin([3, 6])]
    # df['A'].isin([3, 6])

    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(filteredsift, q_path, 100, 0.6, 50)
    a ,d, i_sift, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    print (q_paths.index(q_path), q_path)
    print ('Accuracy =',  a, '%', '| Quality:', d )
    print ('Count', cnt, ' | position', i_sift)

    row_dict['sift_acc'] = a
    row_dict['sift_matchindex'] = i_sift
    row_dict['siftCount'] = cnt
    row_dict['siftquality'] = d
    row_dict['sifttime'] = searchtimesift
    print ("sift processing time", searchtimesift, 'sec')

    # row_dict['hsv' + '_predict20'] = a
    # row_dict['hsv'+ '_quality'] = d
    # row_dict['hsv'+'_time'] = searchtime
    # row_dict['hsv'+'matchposition'] = i_hsv
    # row_dict['rgb'+'matchposition'] = i_rgb
    # row_dict['hsv'+'PCount'] = cnt
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    t = time.time() - start
    row_dict['totaltime'] = t
    print ('total porocessing time = ', t, 'seconds')


    accStatsmerge = accStatsmerge.append( row_dict , ignore_index=True)
# print ('HSV Tree Mean Search time', accStatsmerge['hsv'+'_time'].mean())
# print ("Mean Acc = ", accStatsmerge['hsv' + '_predict20'].mean())

accStatsmerge.to_csv('accStatsmergetestrandsampl1lowe0.6.csv')

