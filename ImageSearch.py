
#------------------------------------HASH CODE-----------------------------------#

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

print('Hello here')

IMGDIR = "./imagesbooks/"
# IMGDIR = "../../images_holidays/jpg/"
# TEST_IMGDIR = "../../test_images/"

haystackPaths = list(paths.list_images(IMGDIR))
# needlePaths = list(paths.list_images(TEST_IMGDIR))
# print(haystackPaths)

# def get_Hash(image, hash_size=32):
#     return imagehash.dhash(image, hash_size=hash_size)


    # init a hash dictionary 
# haystack = {}

# init a hash dataframe
haystack = pd.DataFrame(columns=['file', 'phash', 'ahash', 'dhash', 'whash'])

# time the hashing operation 
start = time.time()

for f in haystackPaths:
    
    image = Image.open(f)
#     imageHash = imagehash.phash(image)
    p = imagehash.phash(image, hash_size=32)
    a = imagehash.average_hash(image, hash_size=32)
    d = imagehash.dhash(image, hash_size=32)
    w = imagehash.whash(image, hash_size=32)

    haystack = haystack.append ({'file':f, 'phash':p, 'ahash':a, 'dhash':d,'whash':w }, ignore_index=True)
print (haystack.head())

#     print (p, imageHash)
    
#     haystack[imageHash] = p

# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# for search images 
 

import pandas as pd 
import random
sample = ['./imagesbooks/ukbench07994.jpg']
# sample = random.sample(haystackPaths, 1)
# sample = ['./images/ukbench00019.jpg', './images/ukbench00025.jpg', './images/ukbench00045.jpg', './images/ukbench00003.jpg', './images/ukbench00029.jpg']
# sample = ['./images/ukbench00048.jpg', './images/ukbench00016.jpg', './images/ukbench00045.jpg']

print (sample)

for f in sample:
    
#     print ("Searching", p)
    image = Image.open(f)

#     hashes = pd.DataFrame(columns=['file', 'phash', 'ahash', 'dhash', 'whash'])

    hashes =  haystack.copy()


#     imageHash = imagehash.phash(image)
    p = imagehash.phash(image, hash_size=32)
    a = imagehash.average_hash(image, hash_size=32)
    d = imagehash.dhash(image, hash_size=32)
    w = imagehash.whash(image, hash_size=32)
    
    hashes['phash']= hashes['phash'] - p
    hashes['ahash']= hashes['ahash'] - a
    hashes['dhash']= hashes['dhash'] - d
    hashes['whash']= hashes['whash'] - w
    print(hashes)

    hashes['phash'].plot()
    plt.show()    
    hashes['dhash'].plot()
    plt.show()    
    hashes['ahash'].plot()
    plt.show()    
    hashes['whash'].plot()
    plt.show()


#     imageHash = get_Hash(image)    
#     needlehash[imageHash] = p
    
#     for key in haystack.keys():
#         h = (key - imageHash)
# #         h = distance.hamming(key, imageHash)
# #         print (h, haystack[key])        
#         # matchhash[key] = h
#         hashes = hashes.append ({'file':haystack[key], 'value':h }, ignore_index=True)
# #     print (df)


    for item in list(['phash','ahash','dhash','whash']):
        top = hashes.sort_values(by=[item])[:20]
        d = list(top['file'])
        p = list(top[item])
        # print (d)


        import numpy as np

        
        fig=plt.figure(figsize=(40, 40))
        columns = 20
        rows = 1
        l = 0
        # ax enables access to manipulate each of subplots
        ax = []


        for i in range(1, columns*rows +1):
            img = plt.imread(d[l])
            ax.append(fig.add_subplot(rows, columns, i))
            ax[-1].set_title('score='+str(p[l]))
            plt.imshow(img)
            l +=1
        plt.show()
        

    
# print ("Done") 
# print(hashes)

# hashes['value'].plot()
# plt.show()

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

top = hashes
d = list(top['file'])
p = list(top[item])
# print(top)

import cv2

def gen_sift_features(image):
    sift = cv2.xfeatures2d.SIFT_create(1000)
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

# FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
sample

# Hyper-Parameters for SIFT comparison
sift_features_limit = 1000
lowe_ratio = 0.75
predictions_count = 10


predictions = []

matches_flann = []
# Reading query image

q_path = random.sample(haystackPaths, 1)[0]
q_img = cv2.imread(q_path)    
q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
# Generating SIFT features for query image
q_kp,q_des = gen_sift_features(q_img)
    

for j in d:
    matches_count = 0
    m_path = j
    m_img = cv2.imread(m_path)        
    if m_img is None:
        continue
    m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
    
    # Generating SIFT features for predicted ssim images
    m_kp,m_des = gen_sift_features(m_img)
    siftdf = siftdf.append({'file':j, 'siftkey':m_kp, 'siftdesc':m_des}, ignore_index=True)
    if m_des is None:
        continue
        
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
print(predictions)

import numpy as np


fig=plt.figure(figsize=(40, 40))
columns = 10
rows = 1
l = 0
# ax enables access to manipulate each of subplots
ax = []


for i in range(1, columns*rows +1):
    img = plt.imread(d[l])
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title('score='+str(p[l]))
    plt.imshow(img)
    l +=1
plt.show()

fig=plt.figure(figsize=(40, 40))
columns = 10
rows = 1
l = 0
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

