'''
SIFT, SURF, ORB are patented and no longer available opencv 4.0 
install last opensource version 

Ref: https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal/52514095

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

#-------------------------SIFT FUNCTIONAL FORMAT---------------------------#

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


#-----------------TRAINING IMAGES SIFT FEATURE GENERATION-----------------------#

def gen_sift_features(imagelibrarypaths, sift_features_limit):

    # init a sift dataframe
    siftdf = pd.DataFrame(columns=['file', 'siftkey', 'siftdes'])

    # time the hashing operation 
    start = time.time()

    
    for f in imagelibrarypaths:

        m_img = cv2.imread(f)        
        if m_img is None:
            continue
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        
       
        sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
        # kp is the keypoints
        # desc is the SIFT descriptors, they're 128-dimensional vectors
        # that we can use for our final features
        kp, desc = sift.detectAndCompute(m_img, None)
        # m_kp,m_des = gen_sift_features(m_img)
        siftdf = siftdf.append({'file':f, 'siftkey':kp, 'siftdes':desc}, ignore_index=True)
        
    t= time.time() - start
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(imagelibrarypaths), t))
    # print (siftdf.head())
    return (siftdf,  t)

'''
Save Pandas dataframe to pickle 
Datafram format : file , imagehist
'''

def SIFT_SAVE_FEATURES ( mydataSIFT, savefile='testSIFT_Data') : 

    # save the tree #example # treeName = 'testSIFT_Data.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( mydataSIFT[['file', 'siftdes']] , outfile)


'''
Load Pandas dataframe from pickle 
Datafram format : file , siftdes
'''
def SIFT_LOAD_FEATURES ( openfile='testSIFT_Data') : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    mydataSIFT = pickle.load(infile)
    infile.close()

    return mydataSIFT


def FEATURE (queryimagepath, sift_features_limit=100):
    # start = time.time()
    q_img = cv2.imread(queryimagepath)    
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    q_kp, q_des = sift.detectAndCompute(q_img, None)

    return q_kp, q_des


    
#------------------QUERY IMAGE FEATURE GEN---------------#

def SIFT_SEARCH (feature, queryimagepath, sift_features_limit=100, lowe_ratio=0.75, predictions_count=50):
    start = time.time()
    q_img = cv2.imread(queryimagepath)    
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    q_kp, q_des = sift.detectAndCompute(q_img, None)
   
   
    # FLANN matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    
    matches_flann = []

    for index, j in feature.iterrows(): 
        m_des = j['siftdes'] 
        m_path = j['file']     
        # Calculating number of feature matches using FLANN
        matches = flann.knnMatch(q_des,m_des,k=2)

        #ratio query as per Lowe's paper
        matches_count = 0
        for x,(m,n) in enumerate(matches):
            if m.distance < lowe_ratio*n.distance:
                matches_count += 1
        matches_flann.append((matches_count,m_path))


    matches_flann.sort(key=lambda x : x[0] , reverse = True)
    predictions = matches_flann[:predictions_count]
    t= time.time() - start
    # print(predictions)
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(haystackPaths), t))
    return (predictions, t)
    ## Search End


def SIFT_SEARCH_BF (feature, queryimagepath, sift_features_limit=100, lowe_ratio=0.75, predictions_count=50):
    start = time.time()
    q_img = cv2.imread(queryimagepath)    
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    q_kp, q_des = sift.detectAndCompute(q_img, None)
   

    # BF macher 
    bf = cv2.BFMatcher()

    matches_BF = []

    for index, j in feature.iterrows():
        m_des = j['siftdes']
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


def SIFT_CREATE_TREE_MODEL ( mydataSIFT, savefile='testSIFTtree', n_clusters=500 ) : 

    print ("Generating SIFT Clusters BOvW and SIFTtree")
    ### Train KMeans and define Feature Vectors 
    # define cluster size 
    # n_clusters = 5000
    # Concatenate all descriptors in the training set together
    training_descs = list(mydataSIFT['siftdes'])
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
    SIFTtree = KDTree(img_bow_hist)
    print('SIFT Tree generation complete.')

    # save the tuple (model, tree)
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( (SIFTtree , cluster_model, img_bow_hist) ,outfile)

    print ('Saved (Tree, Model) as ', outfile)

    return ( SIFTtree , cluster_model, img_bow_hist)


def SIFT_Load_Tree_Model ( openfile='testSIFTtree' ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    SIFTtree, cluster_model, img_bow_hist = pickle.load(infile)
    infile.close()

    return SIFTtree , cluster_model


'''
Creates Cluster from FVHistograms and returns ID
'''
def SIFT_RUN_CLUSTER ( img_bow_hist, mydataSIFT, n_clusters ) : 
    SIFTCluster = KMeans(n_clusters=n_clusters)
    SIFTCluster.fit(img_bow_hist)
    SIFTCluster.predict(img_bow_hist)
    labels = SIFTCluster.labels_
    # print (labels)

    # update labels to original dataframe
    mydataSIFT['clusterID'] = pd.DataFrame(labels)

    return mydataSIFT


def SIFT_ANALYZE_CLUSTER (img_bow_hist, n_clusters=200, step=10) :  
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


def SIFT_SEARCH_TREE (q_path, cluster_model, SIFTtree, mydataSIFT, returnCount=100, kp=100) : 
    print ("searching Tree.")
    # # sample a image
    # q_path = random.sample(imagepaths, 1)[0]
    # print (q_path)
    # log time 
    start = time.time()
    # get the feature for this image 
    q_kp, q_des = FEATURE (q_path , kp)
    # get bow cluster
    q_clustered_words = cluster_model.predict(q_des) 
    # get FV histogram  
    q_bow_hist = np.array([np.bincount(q_clustered_words, minlength=cluster_model.n_clusters)])
    # search the KDTree for nearest match
    dist, result = SIFTtree.query(q_bow_hist, k=returnCount)
    t= time.time() - start
    # Zip results to list of tuples 
    flist = list (mydataSIFT.iloc[ result[0].tolist()]['file'])
    slist = list (dist[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples frm 2 lists

    return (matches, t)




# # ------------ GENERATION TEST-------------------#


# # Hyper-Parameters for SIFT comparison
# sift_features_limit = 1000
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
