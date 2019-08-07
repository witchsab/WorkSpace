'''
SIFT, SURF, ORB are patented and no longer available opencv 4.0 
install last opensource version 

Ref: https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal/52514095

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

#-------------------------ORB FUNCTIONAL FORMAT---------------------------#
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
import pickle


IMGDIR = "./imagesbooks/"


#-----------------TRAINING IMAGES ORB FEATURE GENERATION-----------------------#

def GEN_ORB_FEATURES(imagelibrarypaths, ORB_features_limit):

    # init a ORB dataframe
    ORBdf = pd.DataFrame(columns=['file', 'ORBkey', 'ORBdes'])

    # time the hashing operation
    start = time.time()

    for f in imagelibrarypaths:

        m_img = cv2.imread(f)
        if m_img is None:
            continue
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        # Initiate STAR detector
        orb = cv2.ORB_create(ORB_features_limit)

        # find the keypoints with ORB
        kp = orb.detect(m_img, None)

        # compute the descriptors with ORB
        kp, desc = orb.compute(m_img, kp)

        ORBdf = ORBdf.append(
            {'file': f, 'ORBkey': kp, 'ORBdes': desc}, ignore_index=True)

    t = time.time() - start
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(imagelibrarypaths), t))
    # print (ORBdf.head())
    return (ORBdf,  t)


'''
Save Pandas dataframe to pickle 
Datafram format : file , imagehist
'''


def ORB_SAVE_FEATURES(mydataORB, savefile='testORB_Data'):

    # save the tree #example # treeName = 'testORB_Data.pickle'
    outfile = open(savefile + '.pickle', 'wb')
    pickle.dump(mydataORB[['file', 'ORBdes']], outfile)


'''
Load Pandas dataframe from pickle 
Datafram format : file , ORBdes
'''


def ORB_LOAD_FEATURES(openfile='testORB_Data'):

    # reading the pickle tree
    infile = open(openfile + '.pickle', 'rb')
    mydataORB = pickle.load(infile)
    infile.close()

    return mydataORB


#------------------QUERY IMAGE FEATURE GEN---------------#

def ORB_SEARCH_BF (feature, queryimagepath, ORB_features_limit=100, lowe_ratio=0.7, predictions_count=50):
    start = time.time()
    q_img = cv2.imread(queryimagepath)
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    ORB = cv2.ORB_create(ORB_features_limit)
    q_kp, q_des = ORB.detectAndCompute(q_img, None)



    # BF macher 
    bf = cv2.BFMatcher()

    matches_BF = []

    for index, j in feature.iterrows():
        m_des = j['ORBdes']
        m_path = j['file']
        # Calculating number of feature matches using FLANN
        matches = bf.knnMatch(q_des, m_des, k=2)

        # ratio query as per Lowe's paper
        matches_count = 0
        for x, (m, n) in enumerate(matches):
            if m.distance < lowe_ratio*n.distance:
                matches_count += 1
        matches_BF.append((matches_count, m_path))

    matches_BF.sort(key=lambda x: x[0], reverse=True)
    predictions = matches_BF[:predictions_count]
    t = time.time() - start
    # print(predictions)
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(haystackPaths), t))
    return (predictions, t)
    # Search End


# https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c
def ORB_SEARCH_FLANN(feature, queryimagepath, ORB_features_limit=100, lowe_ratio=0.7, predictions_count=50):
    start = time.time()
    q_img = cv2.imread(queryimagepath)
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    ORB = cv2.ORB_create(ORB_features_limit)
    q_kp, q_des = ORB.detectAndCompute(q_img, None)


    # FLANN matcher
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2


    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    
    matches_flann = []

    for index, j in feature.iterrows(): 
        m_des = j['ORBdes'] 
        m_path = j['file']     
        # Calculating number of feature matches using FLANN
        matches = flann.knnMatch(q_des,m_des,k=2)

        #ratio query as per Lowe's paper
        matches_count = 0
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio:
                matches_count += 1
        matches_flann.append((matches_count,m_path))


    matches_flann.sort(key=lambda x: x[0], reverse=True)
    predictions = matches_flann[:predictions_count]
    t = time.time() - start
    # print(predictions)
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(haystackPaths), t))
    return (predictions, t)
    # Search End


# Do not use yet 
def ORB_SEARCH_MODBF(feature, queryimagepath, ORB_features_limit=100, lowe_ratio=0.7, predictions_count=50):
    start = time.time()
    q_img = cv2.imread(queryimagepath)
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    ORB = cv2.ORB_create(ORB_features_limit)
    q_kp, q_des = ORB.detectAndCompute(q_img, None)

    # bf = cv2.BFMatcher ()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_BF = []

    for index, j in feature.iterrows():
        m_des = j['ORBdes']
        m_path = j['file']
        # Calculating number of feature matches using FLANN
        matches = bf.match(q_des, m_des)

        # ratio query as per Lowe's paper
        matches_count = 0
        for m in matches:
            if m.distance < lowe_ratio:
                matches_count += 1
        # matches_BF.append((matches_count, m_path))
        matches_BF.append((len(matches), m_path))

    matches_BF.sort(key=lambda x: x[0], reverse=True)
    predictions = matches_BF[:predictions_count]
    t = time.time() - start
    # print(predictions)
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(haystackPaths), t))
    return (predictions, t)
    # Search End


# # ------------ GENERATION TEST-------------------#


# # Hyper-Parameters for ORB comparison
# ORB_features_limit = 1000
# lowe_ratio = 0.75
# predictions_count = 50

# IMGDIR = "./imagesbooks/"
# imagepaths = list(paths.list_images(IMGDIR))
# mydata1, mytime1 = gen_sift_features(imagepaths, 1000)


# # ------------------ SEARCH TEST ---------------------#

# q_path = random.sample(imagepaths, 1)[0]
# imagepredictions , searchtime = SIFT_SEARCH(mydata1, q_path, 300,0.75, 50)

# # to reload module: uncomment use the following
# # %load_ext autoreload
# # %autoreload 2

# import Accuracy as accuracy
# a = accuracy.accuracy_matches(q_path, imagepredictions[:20], 50)
# print ('Accuracy =',  a, '%')

# import ImageSearch_Plots as myplots
# myplots.plot_predictions(imagepredictions[:20], q_path)


# #---------------- Compile data and plot results

# accStats = pd.DataFrame(columns=['file','Acc', 'PCount'])

# q_paths = random.sample(imagepaths, 50)  # random sample 100 items in list

# for q_path in q_paths:
#     print ("Processing, time", q_paths.index(q_path), searchtime)
#     imagepredictions , searchtime = SIFT_SEARCH(mydata1, q_path, 1000,0.75, 50)
#     a = accuracy.accuracy_matches(q_path, imagepredictions, 50)
#     accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': len(imagepredictions) } , ignore_index=True)


# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStats['Acc'].mean())


# # ------------------ PLOT/DISPLAY RESULTS --------------------
# fig=plt.figure(figsize=(40, 40))
# columns = 5
# rows = 10
# l = 0
# # ax enables access to manipulate each of subplots
# ax = []

# mylist = imagepredictions
# for i in range(1, columns*rows +1):
#     b,a = mylist [l]
#     img = plt.imread(a)
#     ax.append(fig.add_subplot(rows, columns, i))
#     ax[-1].set_title('score='+str(b))
#     plt.imshow(img)
#     l +=1
# plt.show()


# sift_score = pd.DataFrame (columns=['score'])
# for key in mylist:
#     b, a = key
#     sift_score = sift_score.append(
#         {
#             'score' : b
#         }, ignore_index=True
#     )

# sift_score.plot()
# plt.show()
