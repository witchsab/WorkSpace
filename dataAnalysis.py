
import os
import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths

# for hash all the images in folder / database
import Accuracy as accuracy
import ImageSearch_Algo_Hash
import ImageSearch_Algo_HSV
import ImageSearch_Algo_HSV_Fast
import ImageSearch_Algo_ORB
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import ImageSearch_Plots as myplots

# --------------- VAR COMMONS------------------

IMGDIR = r'./imagesbooks/'
IMGDIR = r"V:\\Download\\imagesbooks\\"
IMGDIRPROCESSED = ['']*5
IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"

##############################################################################################


# -------------HSV RGENERATION TEST-------------------#


imagepaths = list(paths.list_images(IMGDIR))
# print (imagepaths)

mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print('HSV Feature Generation time', mytime)


#------------ HSV SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]

# imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSV, q_path)
imagematches, searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSV, q_path)
print('HSV Search time', searchtime)

# to reload module: uncomment use the following
# %load_ext autoreload
# %autoreload 2

a, m, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', ' | Quality: ', m)


# ----- Alternative tree search code [Optimized search time ]

# test TREE SEARCH code

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_HSV.HSV_Create_Tree(mydataHSV, savefile='HSV_Tree')

# to load an existing tree
thistree = ImageSearch_Algo_HSV.HSV_Load_Tree('HSV_Tree')

# sample 1 image 
q_path = random.sample(imagepaths, 1)[0]

imagematches, searchtime = ImageSearch_Algo_HSV.HSV_SEARCH_TREE(
    thistree, mydataHSV, q_path, 50)
print('HSV Tree Search time', searchtime)

a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)


# ------ End Alternative

# HSV Clusters 
n_clusters = 50 
savefile = 'data/' + 'test' + '_HSV_Cluster' + str(n_clusters)
cluster = ImageSearch_Algo_HSV.HSV_CREATE_CLUSTER (mydataHSV, savefile, n_clusters)
md = ImageSearch_Algo_HSV.HSV_RUN_CLUSTER(cluster, mydataHSV)
md.sort_values('file')[['file', 'clusterID']]


# RGB Clusters 
n_clusters = 50 
savefile = 'data/' + 'test' + '_RGB_Cluster' + str(n_clusters)
cluster = ImageSearch_Algo_RGB.RGB_CREATE_CLUSTER (mydataRGB, savefile, n_clusters)
md = ImageSearch_Algo_RGB.RGB_RUN_CLUSTER(cluster, mydataRGB)
md.sort_values('file')[['file', 'clusterID']]


# plot results
myplots.plot_predictions(imagematches[:20], q_path)


##############################################################################################


# -------------RGB RGENERATION TEST-------------------#


# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70


imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataRGB, mytime = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print('RGB Feature Generation time', mytime)

# ImageSearch_Algo_RGB.RGB_SAVE_FEATURES (mydataRGB, 'testRGBPandas')
# loadeddataRGB = ImageSearch_Algo_RGB.RGB_LOAD_FEATURES ('testRGBPandas')

#------------ RGB SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]


# test
# ft = ImageSearch_Algo_RGB.RGB_FEATURE (q_path)

imagematches, searchtime = ImageSearch_Algo_RGB.RGB_SEARCH(
    mydataRGB, q_path, 0.5)
print('RGB Search time', searchtime)

# # to reload module: uncomment use the following
# %load_ext autoreload
# %autoreload 2

a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)


# ----- Alternative tree search code [Optimized search time ]

# test TREE SEARCH code

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_RGB.RGB_Create_Tree (mydataRGB, savefile='RGB_Tree')

# to load an existing tree
thistree = ImageSearch_Algo_RGB.RGB_Load_Tree('RGB_Tree')

# sample 1 image
q_path = random.sample(imagepaths, 1)[0]

imagematches, searchtime = ImageSearch_Algo_RGB.RGB_SEARCH_TREE(
    thistree, mydataRGB, q_path, 100)
print('RGB Tree Search time', searchtime)

a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)


# ------ End Alternative

myplots.plot_predictions(imagematches[:20], q_path)

# ---------------- Compile data and plot results


q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list

accStats = pd.DataFrame(columns=['file', 'Acc', 'PCount', 'Stime'])
for q_path in q_paths:
    imagematches, searchtime = ImageSearch_Algo_RGB.RGB_SEARCH(
        mydataRGB, q_path, 0.5)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append(
        {'file': q_path, 'Acc': a, 'PCount': cnt, 'Stime': searchtime}, ignore_index=True)


plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print("RGB Mean Acc = ", accStats['Acc'].mean(), '%')
print("RGB Mean Search Time = ", accStats['Stime'].mean(), ' secs')



# ----------------- HASH ALGO TESTING CODE----------------------------



# IMGDIR = IMGDIRPROCESSED[3]
# IMGDIR = "./imagesbooks/"
# IMGDIR = "../../images_holidays/jpg/"
# TEST_IMGDIR = "../../test_images/"

imagepaths = list(paths.list_images(IMGDIR))

mydataHASH, mytime = ImageSearch_Algo_Hash.HASH_GEN(imagepaths, 16)
print("HASH All Feature Generation time :", mytime)


# search images

q_path = r'V:\\Download\\imagesbooks\\ukbench00000.jpg'

q_path = random.sample(imagepaths, 1)[0]
testAlgo = 'whash'

# sample = ['./images/ukbench00019.jpg', './images/ukbench00025.jpg', './images/ukbench00045.jpg', './images/ukbench00003.jpg', './images/ukbench00029.jpg']
# sample = ['./images/ukbench00048.jpg', './images/ukbench00016.jpg', './images/ukbench00045.jpg']

#  test on a sample
imagematches, mytime = ImageSearch_Algo_Hash.HASH_SEARCH(q_path, mydataHASH, 100, testAlgo, 16)
# mydata, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (sample, features, 20, 'dhash', 32)
# mydata, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (sample, features, 20, 'ahash', 32)
# mydata, mytime = ImageSearch_Algo_Hash.HASH_SEARCH (sample, features, 20, 'whash', 32)
print(q_path)



a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print("HASH Search time :", mytime)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)


# -----------------HASH test on 100 statistical sample ---------------

q_paths = random.sample(imagepaths, 200)  # random sample 100 items in list

testAlgo = 'phash'

accStats = pd.DataFrame(columns=['file', 'Acc', 'PCount', 'Stime'])
for q_path in q_paths:
    imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH(q_path, mydataHASH, 100, testAlgo, 16)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({'file': q_path, 'Acc': a, 'PCount': cnt,  'Stime': searchtime}, ignore_index=True)

# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print("Mean Acc = ", accStats['Acc'].mean(), '%')
print("Mean Search Time = ", accStats['Stime'].mean(), ' secs')


# -------------------------- Alternative HASH tree search code [Optimized search time ]
testAlgo = 'whash'

# test TREE SEARCH code

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_Hash.HASH_Create_Tree(mydataHASH, 'myHASH_Tree', testAlgo)

# to load an existing tree
# thistree = ImageSearch_Algo_RGB.RGB_Load_Tree('RGB_Tree')

# test over a single sample 
# q_path = random.sample(imagepaths, 1)[0]
imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_TREE(mytree, mydataHASH, q_path, testAlgo, 16, 100)
print('HASH Tree Search time', searchtime)

print ('Test Algo : ', testAlgo)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# -----------------HASH Tree test on 100 statistical sample ---------------

accStats = pd.DataFrame(columns=['file', 'Acc', 'PCount', 'Stime'])
for q_path in q_paths:
    imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_TREE(mytree, mydataHASH, q_path, testAlgo, 16, 100)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({'file': q_path, 'Acc': a, 'PCount': cnt,  'Stime': searchtime}, ignore_index=True)

# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print("Mean Acc = ", accStats['Acc'].mean(), '%')
print("Mean Search Time = ", accStats['Stime'].mean(), ' secs')


# --------------- HYBRID HASH TREE TEST -----------------------------
testAlgoList = ['whash', 'ahash', 'dhash']

# to create a new tree from dataframe features 'mydataHSV'
myHybridtree = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, './data/myHHASH_Tree', testAlgoList)

# test over a single sample 
# q_path = random.sample(imagepaths, 1)[0]
imagematches, searchtime = ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE (myHybridtree, mydataHASH, q_path, testAlgoList, 16, 100)
print('HASH Hybrid Tree Search time', searchtime)

a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# -----------------HYBRID HASH Tree test on 100 statistical sample ---------------

accStats = pd.DataFrame(columns=['file', 'Acc', 'PCount', 'Stime'])
q_paths = imagepaths # random.sample(imagepaths, 200)  # random sample 100 items in list

for q_path in q_paths:
    imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE (myHybridtree, mydataHASH, q_path, testAlgoList, 16, 100)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({'file': q_path, 'Acc_HHash': a, 'Count_HHash_': cnt,  'STime_HHash': searchtime, 'Index_HHash': pos}, ignore_index=True)

# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print("Mean Acc = ", accStats['Acc_HHash'].mean(), '%')
print("Mean Search Time = ", accStats['STime_HHash'].mean(), ' secs')

# ----------------END HYBRID TREE SEARCH STASTICAL DATA COLLECTION ----------------------------

# --------------- ALL COMBINATIONS HYBRID HASH TREE TEST -----------------------------
IMGDIR = r'./imagesbooks/'
imagepaths = sorted(list(paths.list_images(IMGDIR)))
# random.shuffle(imagepaths)
mydataHASH, mytime = ImageSearch_Algo_Hash.HASH_GEN(imagepaths, 16)
print("HASH All Feature Generation time :", mytime)

accStats = pd.DataFrame(columns=['file'])
q_paths = imagepaths # random.sample(imagepaths, 200)  # random sample 100 items in list
testAlgoList = ['ahash','dhash','whash','phash']
for algo in testAlgoList: 
    mytree = ImageSearch_Algo_Hash.HASH_Create_Tree(mydataHASH, './data/myHHASH_'+ str(algo)+'_Tree', algo)
    for q_path in q_paths:
        imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_TREE ( mytree, mydataHASH, q_path, algo, 16, 100)
        a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
        accStats = accStats.append({'file': q_path, 'Acc_HASH_'+str(algo): a, 'Count_HASH_'+str(algo): cnt,  'STime_HASH_'+str(algo): searchtime, 'Index_HASH_'+str(algo): pos}, ignore_index=True)
    print(str(algo),"Samples=",len(q_paths), "Mean Acc = ", accStats['Acc_HASH_'+str(algo)].mean(), '%')
    print(str(algo), "Mean Search Time = ", accStats['STime_HASH_'+str(algo)].mean(), ' secs')

    # accStats.set_index('file', inplace=True)
    accStats.to_csv('./data/test_hash_'+str(algo)+'.csv')


# all combinations of whash 
IMGDIR = r'./imagesbooks/'
imagepaths = sorted(list(paths.list_images(IMGDIR)))
random.shuffle(imagepaths)
mydataHASH, mytime = ImageSearch_Algo_Hash.HASH_GEN(imagepaths, 16)
print("HASH All Feature Generation time :", mytime)

accStats = pd.DataFrame(columns=['file'])
AlgoList = ['whash', 'ahash', 'dhash', 'phash']
hybridlist1 = ['whash', 'ahash', 'dhash']
hybridlist2 = ['whash', 'ahash']
hybridlist3 = ['whash', 'dhash']

mytree = {}
for algo in AlgoList: 
    mytree[algo] = ImageSearch_Algo_Hash.HASH_Create_Tree(mydataHASH, './data/myHHASH_'+ str(algo)+'_Tree', algo)
myHybridtree1 = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, './data/myHHASH_WAD0_Tree', hybridlist1)
myHybridtree2 = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, './data/myHHASH_WA00_Tree', hybridlist2)
myHybridtree3 = ImageSearch_Algo_Hash.HASH_CREATE_HYBRIDTREE(mydataHASH, './data/myHHASH_WD00_Tree', hybridlist3)

for q_path in q_paths:
    row_dict = {'file':q_path}
    for algo in AlgoList: 
        imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_TREE(mytree[algo], mydataHASH, q_path, algo, 16, 100)
        a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)     
        row_dict['acc_HASH_'+str(algo)] = a
        row_dict['index_HASH_'+str(algo)] = pos
        row_dict['Count_HASH_'+str(algo)] = cnt
        row_dict['quality_HASH_'+str(algo)] = q
        row_dict['time_HASH_'+str(algo)] = searchtime

    imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE(myHybridtree1, mydataHASH, q_path, hybridlist1, 16, 100)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['acc_HHASH1'] = a
    row_dict['index_HHASH1'] = pos
    row_dict['Count_HHASH1'] = cnt
    row_dict['quality_HHASH1'] = q
    row_dict['time_HHASH1'] = searchtime
    imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE(myHybridtree2, mydataHASH, q_path, hybridlist2, 16, 100)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['acc_HHASH2'] = a
    row_dict['index_HHASH2'] = pos
    row_dict['Count_HHASH2'] = cnt
    row_dict['quality_HHASH2'] = q
    row_dict['time_HHASH2'] = searchtime
    imagematches, searchtime =  ImageSearch_Algo_Hash.HASH_SEARCH_HYBRIDTREE(myHybridtree3, mydataHASH, q_path, hybridlist3, 16, 100)
    a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
    row_dict['acc_HHASH3'] = a
    row_dict['index_HHASH3'] = pos
    row_dict['Count_HHASH3'] = cnt
    row_dict['quality_HHASH3'] = q
    row_dict['time_HHASH3'] = searchtime

    accStats = accStats.append(row_dict, ignore_index=True)

accStats.to_csv('./data/HASH_519_ALL_Hybrid_TEST.csv')







##############################################################################################


# -------------HSV FAST RGENERATION TEST-------------------#


imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSVFast, mytime = ImageSearch_Algo_HSV_Fast.HSV_GEN(imagepaths)
print('RGB Feature Generation time', mytime)


#------------ HSV FAST SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]

# imagematches , searchtime = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSVFast, q_path)
imagematches, searchtime = ImageSearch_Algo_HSV_Fast.HSV_SEARCH(
    mydataHSVFast, q_path, 0.5)
print('HSV Search time', searchtime)


# to reload module: uncomment use the following
# %load_ext autoreload
# %autoreload 2

a = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%')


myplots.plot_predictions(imagematches[:20], q_path)

# ---------------- HSV FAST Compile data and plot results


q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list

accStats = pd.DataFrame(columns=['file', 'Acc', 'PCount', 'Stime'])
for q_path in q_paths:
    imagematches, searchtime = ImageSearch_Algo_HSV_Fast.HSV_SEARCH(
        mydataHSVFast, q_path, 0.5)
    a = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({'file': q_path, 'Acc': a, 'PCount': len(
        imagematches), 'Stime': searchtime}, ignore_index=True)


plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print("RGB Mean Acc = ", accStats['Acc'].mean(), '%')
print("RGB Mean Search Time = ", accStats['Stime'].mean(), ' secs')


# --------------------------------- ORB GENERATION TEST-------------------#

# Hyper-Parameters for ORB comparison
ORB_features_limit = 100
lowe_ratio = 0.75
predictions_count = 50

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
mydataORB, mytime1 = ImageSearch_Algo_ORB.GEN_ORB_FEATURES(imagepaths, ORB_features_limit)
print("ORB Feature Generation time :", mytime1)


# THE SEARCH ---------

q_path = random.sample(imagepaths, 1)[0]
print(q_path)

# FLANN Macher (fastest) (non repeatable)
imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_FLANN(mydataORB, q_path, ORB_features_limit , 0.7, 50 )
print (" ORB Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# BF Macher (slowest)
imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_BF (mydataORB, q_path, ORB_features_limit , 0.7, 50 )
print (" ORB Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# Modified BF matcher (faster)
imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_MODBF (mydataORB, q_path, ORB_features_limit , 0.7, 50 )
print (" ORB Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# plot the results 
# myplots.plot_predictions(imagematches[:20], q_path)


# ----- Alternative tree search code [Optimized search time ]

# test ORB TREE SEARCH code
ORB_features_limit = 200
lowe_ratio = 0.75
predictions_count = 50
n_clusters = 500
savefile = 'data/' + TESTNAME + '_ORB_Tree_Cluster' + str(n_clusters)
ORBtree, ORBmodel = ImageSearch_Algo_ORB.ORB_CREATE_TREE_MODEL(mydataORB, savefile, n_clusters)

# to load an existing tree
thistree, thismodel = ImageSearch_Algo_ORB.ORB_Load_Tree_Model (savefile)

# sample 1 image 
q_path = random.sample(imagepaths, 1)[0]

# Run Tree Search 
imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_TREE(q_path, thismodel, thistree, mydataORB, returnCount=100, kp=ORB_features_limit)
print('ORB Tree Search time', searchtime)

a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# FLANN-LSH Macher (fastest) (non repeatable)
imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_FLANN(mydataORB, q_path, ORB_features_limit , 0.7, 50 )
print ("ORB FLANN-LSH Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# BF Macher (slowest)
imagematches, searchtime = ImageSearch_Algo_ORB.ORB_SEARCH_BF (mydataORB, q_path, ORB_features_limit , 0.7, 50 )
print ("ORB BF Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# ----- END Alternative 




# compiled run over 100 samples 
q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list

accStatssift = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file' : q_path }
    
    imagepredictions , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH (mydataSIFT, q_path, sift_features_limit, 0.75, 50)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )

    # adding to dict 
    row_dict['kp_' + str(sift_features_limit) + '_predict10'] = a
    row_dict['kp_'+str(sift_features_limit)+'_quality'] = d
    row_dict['kp_'+str(sift_features_limit)+'_time'] = searchtime
    row_dict['kp_'+str(sift_features_limit)+'matchposition'] = i
    row_dict['kp_'+str(sift_features_limit)+'PCount'] = cnt

    accStatssift = accStatssift.append( row_dict , ignore_index=True)
    print ("Processing, time", q_paths.index(q_path), searchtime)


# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['kp_100_time'].mean(), 0, 100, 'r')

print ("Mean Accuracy= ", accStatssift['kp_100_predict10'].mean())
print ("Mean Quality = ", accStatssift['kp_100_quality'].mean())
print ("Mean time    = ", accStatssift['kp_100_time'].mean())
print ("Mean count   = ", accStatssift['hsvPCount'].mean())












# --------------------------------SIFT GENERATION TEST-------------------#

# Hyper-Parameters for SIFT comparison
sift_features_limit = 100
lowe_ratio = 0.7
predictions_count = 50

# IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
mydataSIFT, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(
    imagepaths, sift_features_limit)
print("SIFT Feature Generation time :", mytime1)


# Method 1
# save to pickle : descriptors only; keypoints can be pickled directly
# save the tree #example # treeName = 'testRGB.pickle'
savefile = 'SIFT_features_519set_100_pandas'
outfile = open(savefile + '.pickle', 'wb')
pickle.dump(mydataSIFT[['file', 'siftdes']], outfile)
# note: cv2.keypoints cant be pickled directly

# Method 2
# save to pandas datastore file - not a good idea
hdfSIFT = pd.HDFStore('SIFT_Features.h5')
hdfSIFT.put('mydataSIFT', mydataSIFT[['file', 'siftdes']], data_columns=True)

## Consolidated 
savefile = 'data/SIFT_features_519set_kp100_pandas'
ImageSearch_Algo_SIFT.SIFT_SAVE_FEATURES (mydataSIFT, savefile)
mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES(savefile)




# ------------------SIFT  SEARCH TEST ---------------------#

q_path = random.sample(imagepaths, 1)[0]

# FLANN Macher (slower)
# imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydataSIFT, q_path, sift_features_limit , 0.6, 50 )

# BF Macher faster 
imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydataSIFT, q_path, sift_features_limit , 0.6, 50 )

print (" SIFT Search time :", searchtime)
print(q_path)

# # to reload module: uncomment use the following
# %load_ext autoreload
# %autoreload 2


a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)



# ----- Alternative tree search code [Optimized search time ]

# test SIFT TREE SEARCH code
n_clusters = 500
savefile = 'data/' + 'test' + '_SIFT_Tree_Cluster' + str(n_clusters)
SIFTtree, SIFTmodel = ImageSearch_Algo_SIFT.SIFT_CREATE_TREE_MODEL(mydataSIFT, savefile, n_clusters)

# to load an existing tree
thistree, thismodel = ImageSearch_Algo_SIFT.SIFT_Load_Tree_Model (savefile)

# sample 1 image 
q_path = random.sample(imagepaths, 1)[0]

# Run Tree Search 
imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_TREE(q_path, thismodel, thistree, mydataSIFT, returnCount=100, kp=100)
print('SIFT Tree Search time', searchtime)

a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# BF Macher faster 
imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH_BF(mydataSIFT, q_path, sift_features_limit , 0.6, 50 )
print (" SIFT BF Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)


# FLANN Macher (slower)
imagematches, searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydataSIFT, q_path, sift_features_limit , 0.6, 50 )
print (" SIFT FLANN Search time :", searchtime)
a, q, pos, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print('Accuracy =',  a, '%', '| Quality:', q)
print('Count', cnt, ' | position', pos)

# ----- END Alternative 



# plot the results 
# myplots.plot_predictions(imagematches[:20], q_path)


# compiled run over 100 samples 
q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list

accStatssift = pd.DataFrame(columns=['file'])

for q_path in q_paths: 
    row_dict = {'file' : q_path }
    
    imagepredictions , searchtime = ImageSearch_Algo_SIFT.SIFT_SEARCH (mydataSIFT, q_path, sift_features_limit, 0.75, 50)
    a, d, i, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )

    # adding to dict 
    row_dict['kp_' + str(sift_features_limit) + '_predict10'] = a
    row_dict['kp_'+str(sift_features_limit)+'_quality'] = d
    row_dict['kp_'+str(sift_features_limit)+'_time'] = searchtime
    row_dict['kp_'+str(sift_features_limit)+'matchposition'] = i
    row_dict['kp_'+str(sift_features_limit)+'PCount'] = cnt

    accStatssift = accStatssift.append( row_dict , ignore_index=True)
    print ("Processing, time", q_paths.index(q_path), searchtime)


# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['kp_100_time'].mean(), 0, 100, 'r')

print ("Mean Accuracy= ", accStatssift['kp_100_predict10'].mean())
print ("Mean Quality = ", accStatssift['kp_100_quality'].mean())
print ("Mean time    = ", accStatssift['kp_100_time'].mean())
print ("Mean count   = ", accStatssift['hsvPCount'].mean())

accStatssift.to_csv('data/accStatssift_kp100_query50_poolAll.csv')
