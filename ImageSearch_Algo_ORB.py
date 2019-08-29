'''
ORB, SURF, ORB are patented and no longer available opencv 4.0 
install last opensource version 

Ref: https://stackoverflow.com/questions/52305578/ORB-cv2-xfeatures2d-ORB-create-not-working-even-though-have-contrib-instal/52514095

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

#-------------------------ORB FUNCTIONAL FORMAT---------------------------#

import os
import pickle
import random
import time
from pprint import pprint

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from imutils import paths
from kneed import KneeLocator
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KDTree

# IMGDIR = "./imagesbooks/"


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
        if len(kp) < 1:
            desc = np.zeros((1, orb.descriptorSize()), np.float32)    
        
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
    pickle.dump(mydataORB[['file', 'ORBkey', 'ORBdes']], outfile)


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


def FEATURE (queryimagepath, ORB_features_limit=100):
    # start = time.time()
    q_img = cv2.imread(queryimagepath)    
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    ORB = cv2.ORB_create(ORB_features_limit)
    q_kp, q_des = ORB.detectAndCompute(q_img, None)

    return q_kp, q_des


#------------------QUERY IMAGE FEATURE GEN---------------#

def ORB_SEARCH_BF (feature, queryimagepath, ORB_features_limit=100, lowe_ratio=0.7, predictions_count=50):
    start = time.time()


    # if isNew: 
    #     # Create Features for new Images
    #     q_img = cv2.imread(queryimagepath)    
    #     q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    #     sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    #     q_kp, q_des = sift.detectAndCompute(q_img, None)
    # else: 
    # Search within the feature for known
    q_des = np.vstack(feature[feature['file'] == queryimagepath]['ORBdes'])
    

    # BF macher 
    bf = cv2.BFMatcher()

    matches_BF = []

    for index, j in feature.iterrows():
        m_des = j['ORBdes']
        m_path = j['file']
        
        try:  
            # Calculating number of feature matches using FLANN
            matches = bf.knnMatch(q_des, m_des, k=2)

            # ratio query as per Lowe's paper
            matches_count = 0
            for x, (m, n) in enumerate(matches):
                if m.distance < lowe_ratio*n.distance:
                    matches_count += 1
            matches_BF.append((matches_count, m_path))
        except: 
            print ('ORB ERROR', m_path, 'qDes-Shape: ', q_des.shape, 'm_des-Shape', m_des.shape)  
            # print ('ORB ERROR')
            # print ('Query', q_des)
            # print ('Search', m_des)
            # print ('Index' , index, m_path)
            # print ('BF Match count ', len(matches))

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

# if isNew: 
    #     # Create Features for new Images
    #     q_img = cv2.imread(queryimagepath)    
    #     q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    #     sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    #     q_kp, q_des = sift.detectAndCompute(q_img, None)
    # else: 
    # Search within the feature for known
    q_des = np.vstack(feature[feature['file'] == queryimagepath]['ORBdes'])
    


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
        
        try: 
            # Calculating number of feature matches using FLANN
            matches = flann.knnMatch(q_des,m_des,k=2)

            #ratio query as per Lowe's paper
            matches_count = 0
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio:
                    matches_count += 1
            matches_flann.append((matches_count,m_path))
        except: 
            print ('ORB ERROR', m_path, 'qDes-Shape: ', q_des.shape, 'm_des-Shape', m_des.shape)  
            # print ('ORB ERROR')            
            # print ('Query', q_des)
            # print ('Search', m_des)
            # print ('Index' , index, m_path)
            # print ('BF Match count ', len(matches))

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

    # if isNew: 
    #     # Create Features for new Images
    #     q_img = cv2.imread(queryimagepath)    
    #     q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    #     sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    #     q_kp, q_des = sift.detectAndCompute(q_img, None)
    # else: 
    # Search within the feature for known
    q_des = np.vstack(feature[feature['file'] == queryimagepath]['ORBdes'])
    

    # bf = cv2.BFMatcher ()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_BF = []

    for index, j in feature.iterrows():
        m_des = j['ORBdes']
        m_path = j['file']
        
        try:
            # Calculating number of feature matches using FLANN
            matches = bf.match(q_des, m_des)

            # ratio query as per Lowe's paper
            matches_count = 0
            for m in matches:
                if m.distance < lowe_ratio:
                    matches_count += 1
            # matches_BF.append((matches_count, m_path))
            matches_BF.append((len(matches), m_path))
        except: 
            print ('ORB ERROR', m_path, 'qDes-Shape: ', q_des.shape, 'm_des-Shape', m_des.shape)  
            # print ('ORB ERROR')
            # print ('Query', q_des)
            # print ('Search', m_des)
            # print ('Index' , index, m_path)
            # print ('BF Match count ', len(matches))

    matches_BF.sort(key=lambda x: x[0], reverse=True)
    predictions = matches_BF[:predictions_count]
    t = time.time() - start
    # print(predictions)
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(haystackPaths), t))
    return (predictions, t)
    # Search End



# ORB TREE GENERATION and SEARCH using BOVW histograms method 

def ORB_CREATE_TREE_MODEL ( mydataORB, savefile='testORBtree', n_clusters=500 ) : 

    print ("Generating ORB Clusters BOvW and ORBtree")
    ### Train KMeans and define Feature Vectors 
    # define cluster size 
    # n_clusters = 5000
    # Concatenate all descriptors in the training set together
    training_descs = list(mydataORB['ORBdes'])
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    # define the cluster model 
    cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print('done clustering. Using clustering model to generate BoW histograms for each image.')
    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in training_descs]
    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array([np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])
    # create Tree for histograms 
    ORBtree = KDTree(img_bow_hist)
    print('ORB Tree generation complete.')

    # save the tuple (model, tree)
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( (ORBtree, cluster_model, img_bow_hist), outfile)

    print ('Saved (Tree, Model) as ', outfile)

    return ( ORBtree , cluster_model, img_bow_hist)


def ORB_Load_Tree_Model ( openfile='testORBtree' ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    ORBtree, cluster_model, img_bow_hist = pickle.load(infile)
    infile.close()

    return (ORBtree , cluster_model, img_bow_hist)


def ORB_SEARCH_TREE (q_path, cluster_model, ORBtree, mydataORB, returnCount=100, kp=100) : 
    # print ("searching Tree.")
    # # sample a image
    # q_path = random.sample(imagepaths, 1)[0]
    # print (q_path)
    # log time 
    start = time.time()

    # # get the feature for this image 
    # if isNew: 
    #     # Create Features for new Images
    #     q_img = cv2.imread(queryimagepath)    
    #     q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    #     sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    #     q_kp, q_des = sift.detectAndCompute(q_img, None)
    # else: 
    # Search within the feature for known
    q_des = np.vstack(mydataORB[mydataORB['file'] == q_path]['ORBdes'])

    # get bow cluster
    q_clustered_words = cluster_model.predict(q_des) 
    # get FV histogram  
    q_bow_hist = np.array([np.bincount(q_clustered_words, minlength=cluster_model.n_clusters)])
    # search the KDTree for nearest match
    dist, result = ORBtree.query(q_bow_hist, k=returnCount)
    t= time.time() - start
    # Zip results to list of tuples 
    flist = list (mydataORB.iloc[ result[0].tolist()]['file'])
    slist = list (dist[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples frm 2 lists

    return (matches, t)

    
# Clustering codes (Advanced Use Only)

'''
Creates Cluster from FVHistograms and returns ID
'''
def ORB_RUN_CLUSTER ( img_bow_hist, mydataORB, n_clusters ) : 
    ORBCluster = KMeans(n_clusters=n_clusters)
    ORBCluster.fit(img_bow_hist)
    ORBCluster.predict(img_bow_hist)
    labels = ORBCluster.labels_
    # print (labels)

    # update labels to original dataframe
    mydataORB['clusterID'] = pd.DataFrame(labels)

    return mydataORB


def ORB_ANALYZE_CLUSTER (img_bow_hist, n_clusters=200, step=10) :  
    print ('n_clusters:', n_clusters, '| step:',  step)
    
    X = img_bow_hist
    # Calculate clusters using Elbow criteria 
    wcss = []
    for i in range(1, n_clusters, step ):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, n_clusters, step), wcss)
    # plt.plot(range(1, 11), elbowIndex)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    # find elbow using Kneelocator package 
    # https://github.com/arvkevi/kneed#find-knee
    
    elbow = KneeLocator( list(range(1,n_clusters, step)), wcss, S=1.0, curve='convex', direction='decreasing')
    print ('Detected Elbow cluster value :', elbow.knee)

    return elbow.knee



# # ------------ GENERATION TEST-------------------#


# # Hyper-Parameters for ORB comparison
# ORB_features_limit = 1000
# lowe_ratio = 0.75
# predictions_count = 50

# IMGDIR = "./imagesbooks/"
# imagepaths = list(paths.list_images(IMGDIR))
# mydata1, mytime1 = gen_ORB_features(imagepaths, 1000)


# # ------------------ SEARCH TEST ---------------------#

# q_path = random.sample(imagepaths, 1)[0]
# imagepredictions , searchtime = ORB_SEARCH(mydata1, q_path, 300,0.75, 50)

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
#     imagepredictions , searchtime = ORB_SEARCH(mydata1, q_path, 1000,0.75, 50)
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


# ORB_score = pd.DataFrame (columns=['score'])
# for key in mylist:
#     b, a = key
#     ORB_score = ORB_score.append(
#         {
#             'score' : b
#         }, ignore_index=True
#     )

# ORB_score.plot()
# plt.show()
