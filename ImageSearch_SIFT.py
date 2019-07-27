'''
SIFT, SURF, ORB are patented and no longer available opencv 4.0 
install last opensource version 

Ref: https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal/52514095

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

#--------------------------------SIFT CODE------------------------------------#

# top = hashes.sort_values(by=['dhash'])[:20]
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
IMGDIR = "./imagesbooks/"

haystackPaths = list(paths.list_images(IMGDIR))
d = haystackPaths


def gen_sift_features(image):
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


# init a sift dataframe
siftdf = pd.DataFrame(columns=['file', 'siftkey', 'siftdesc'])

# time the hashing operation 
start = time.time()


# Hyper-Parameters for SIFT comparison
sift_features_limit = 100
lowe_ratio = 0.75
predictions_count = 50


for j in d:

    m_img = cv2.imread(j)        
    if m_img is None:
        continue
    m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
    
    # Generating SIFT features for predicted ssim images
    m_kp,m_des = gen_sift_features(m_img)
    siftdf = siftdf.append({'file':j, 'siftkey':m_kp, 'siftdesc':m_des}, ignore_index=True)
    if m_des is None:
        continue


print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystackPaths), time.time() - start))


# ------------------ SIFT SEARCH CODE ---------------------

import random


start = time.time()

# FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)


# Reading query image
q_path = random.sample(haystackPaths, 1)[0]
# q_path = './imagesbooks/ukbench06453.jpg'  # sample test 
q_img = cv2.imread(q_path)    
q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)


# Generating SIFT features for query image
q_kp,q_des = gen_sift_features(q_img)
    

# Start Search 
predictions = []
matches_flann = []

for index, j in siftdf.iterrows(): 
    m_des = j['siftdesc'] 
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
predictions.append((q_path,matches_flann[:predictions_count]))
# print(predictions)
print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystackPaths), time.time() - start))

## Search End


# ------------------ PLOT/DISPLAY RESULTS --------------------
print (q_path)
columns = 5
rows = 5
l = 0
size = 2
fig=plt.figure(figsize=(size*columns, size*rows))
# ax enables access to manipulate each of subplots
ax = []

x,mylist = predictions[0]
for i in range(1, columns*rows +1):
    b,a = mylist [l]
    img = plt.imread(a)
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title('score='+str(b))
    plt.imshow(img)
    l +=1
plt.show()




sift_score = pd.DataFrame (columns=['score'])
for key in mylist: 
    b, a = key 
    sift_score = sift_score.append(
        {
            'score' : b
        }, ignore_index=True
    )

sift_score.plot()
plt.show()

