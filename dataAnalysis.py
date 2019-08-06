import os 
# import ImageSearch_Algo_Hash as ImageSearch_Algo_Hash
# import ImageSearch_Algo_RGB as ImageSearch_Algo_RGB

import pandas
from imutils import paths
import random 

# --------------- TEST COMMONS------------------

IMGDIR = r"V:\\Download\\imagesbooks\\"
IMGDIRPROCESSED = ['']*5
IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"


##############################################################################################


# -------------HSV RGENERATION TEST-------------------#

import ImageSearch_Algo_HSV 

imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytime)


#------------ HSV SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]

# imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSV, q_path)
imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSV, q_path)
print ('HSV Search time', searchtime)

# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a, m, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', ' | Quality: ', m)


# ----- Alternative tree search code [Optimized search time ]

# test TREE SEARCH code 

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')

# to load an existing tree 
thistree = ImageSearch_Algo_HSV.HSV_Load_Tree('HSV_Tree')

imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (thistree, mydataHSV, q_path, 50)
print ('HSV Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)


# ------ End Alternative 



# plot results 
import ImageSearch_Plots as myplots
myplots.plot_predictions(imagematches[:20], q_path)




##############################################################################################


# -------------RGB RGENERATION TEST-------------------#


import ImageSearch_Algo_RGB


# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70


imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataRGB, mytime = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print ('RGB Feature Generation time', mytime)



#------------ RGB SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]


# test 
# ft = ImageSearch_Algo_RGB.RGB_FEATURE (q_path)

imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, 0.5)
print ('RGB Search time', searchtime)

# # to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)



# ----- Alternative tree search code [Optimized search time ]

# test TREE SEARCH code 

# to create a new tree from dataframe features 'mydataHSV'
# mytree = ImageSearch_Algo_RGB.RGB_Create_Tree (mydataRGB, savefile='RGB_Tree')

# to load an existing tree 
# thistree = ImageSearch_Algo_RGB.RGB_Load_Tree('RGB_Tree')

imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (thistree, mydataRGB, q_path, 100)
print ('RGB Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)


# ------ End Alternative 

import ImageSearch_Plots as myplots
myplots.plot_predictions(imagematches[:20], q_path)

#---------------- Compile data and plot results 

import matplotlib.pyplot as plt
import pandas as pd

q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 

accStats = pd.DataFrame(columns=['file','Acc', 'PCount', 'Stime'])
for q_path in q_paths:    
    imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, 0.5)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': cnt, 'Stime' : searchtime } , ignore_index=True)


plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print ("RGB Mean Acc = ", accStats['Acc'].mean(), '%')
print ("RGB Mean Search Time = ", accStats['Stime'].mean(), ' secs')



# ------------SIFT GENERATION TEST-------------------#


# Hyper-Parameters for SIFT comparison
sift_features_limit = 100
lowe_ratio = 0.75
predictions_count = 50

# IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIRPROCESSED4))
mydataSIFT, mytime1 = gen_sift_features(imagepaths, 500)
print ("SIFT Feature Generation time :", mytime1)

# ------------------SIFT  SEARCH TEST ---------------------#

q_path = random.sample(imagepaths, 1)[0]
imagepredictions , searchtime = SIFT_SEARCH(mydataSIFT, q_path, 300,0.75, 50)

# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a = accuracy.accuracy_matches(q_path, imagepredictions[:20], 50)
print ('Accuracy =',  a, '%')

import ImageSearch_Plots as myplots
myplots.plot_predictions(imagepredictions[:20], q_path)



# ----------------- HASH ALGO TESTING CODE----------------------------


from imutils import paths
import ImageSearch_Algo_Hash 

# for hash all the images in folder / database 

# IMGDIR = IMGDIRPROCESSED[3]
# IMGDIR = "./imagesbooks/"
# IMGDIR = "../../images_holidays/jpg/"
# TEST_IMGDIR = "../../test_images/"

imagepaths = list(paths.list_images(IMGDIR))

mydataHASH , mytime = ImageSearch_Algo_Hash.HASH_GEN (imagepaths, 16)
print ("HASH All Feature Generation time :", mytime)


# search images 

q_path = r'V:\\Download\\imagesbooks\\ukbench00000.jpg'
# q_path = random.sample(imagepaths, 1)[0]
# sample = ['./images/ukbench00019.jpg', './images/ukbench00025.jpg', './images/ukbench00045.jpg', './images/ukbench00003.jpg', './images/ukbench00029.jpg']
# sample = ['./images/ukbench00048.jpg', './images/ukbench00016.jpg', './images/ukbench00045.jpg']

#  test on a sample 
imagematches, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (q_path, mydataHASH, 100, 'phash', 16)
# mydata, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (sample, features, 20, 'dhash', 32)
# mydata, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (sample, features, 20, 'ahash', 32)
# mydata, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (sample, features, 20, 'whash', 32)
print (q_path)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ("HASH Search time :", mytime)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)



# ----- Alternative tree search code [Optimized search time ]

# test TREE SEARCH code 

# to create a new tree from dataframe features 'mydataHSV'
# mytree = ImageSearch_Algo_RGB.RGB_Create_Tree (mydataRGB, savefile='RGB_Tree')

# to load an existing tree 
# thistree = ImageSearch_Algo_RGB.RGB_Load_Tree('RGB_Tree')

imagematches , searchtime = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (thistree, mydataRGB, q_path, 100)
print ('RGB Tree Search time', searchtime)

import Accuracy as accuracy
a , q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', q )
print ('Count', cnt, ' | position', pos)




# Statistics over 100 sample using tree 




# -----------------HASH test on 100 statistical sample ---------------

import matplotlib.pyplot as plt
import pandas as pd
import random


q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 

accStats = pd.DataFrame(columns=['file','Acc', 'PCount', 'Stime'])
for q_path in q_paths:    
    imagematches , searchtime = ImageSearch_Algo_Hash.HASH_SEARCH (q_path, mydataHASH, 100, 'phash', 32)
    a = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': len(imagematches), 'Stime' : searchtime } , ignore_index=True)


plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print ("Mean Acc = ", accStats['Acc'].mean(), '%')
print ("Mean Search Time = ", accStats['Stime'].mean(), ' secs')




# -------------------------END TESTING----------------------------




##############################################################################################


# -------------HSV FAST RGENERATION TEST-------------------#

import ImageSearch_Algo_HSV_Fast

imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSVFast, mytime = ImageSearch_Algo_HSV_Fast.HSV_GEN(imagepaths)
print ('RGB Feature Generation time', mytime)


#------------ HSV FAST SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]

# imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSVFast, q_path)
imagematches , searchtime = ImageSearch_Algo_HSV_Fast.HSV_SEARCH(mydataHSVFast, q_path, 0.5)
print ('HSV Search time', searchtime)


# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%')


import ImageSearch_Plots as myplots
myplots.plot_predictions(imagematches[:20], q_path)

#---------------- HSV FAST Compile data and plot results 

import matplotlib.pyplot as plt
import pandas as pd

q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 

accStats = pd.DataFrame(columns=['file','Acc', 'PCount', 'Stime'])
for q_path in q_paths:    
    imagematches , searchtime = ImageSearch_Algo_HSV_Fast.HSV_SEARCH ( mydataHSVFast, q_path, 0.5)
    a = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': len(imagematches), 'Stime' : searchtime } , ignore_index=True)


plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print ("RGB Mean Acc = ", accStats['Acc'].mean(), '%')
print ("RGB Mean Search Time = ", accStats['Stime'].mean(), ' secs')
