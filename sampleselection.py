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
import numpy as np
import random

IMGDIR = "./imagesbooks/"
# ----------random sampling 20 images------------#
trainhistpath = list(paths.list_images(IMGDIR))
# print (trainhistpath)
Trainhist = pd.DataFrame(columns=['file','imagehist'])
# print (Trainhist.head())
sample1 = random.sample(trainhistpath, 10)
# print (sample)

#--------printing 10 random sample------#
    
# fig=plt.figure(figsize=(40, 40))
# columns = 2
# rows = 5
# l = 0
# ax = []
# d = list(sample1)
# # print(d)

# for i in range(1, columns*rows +1):
#     img = plt.imread(d[l])
#     ax.append(fig.add_subplot(rows, columns, i))
#     ax[-1].set_title(str(d[l]))
#     plt.imshow(img)
#     l +=1
# plt.show()
#-------------printing histogram of 10 samples-------------------#

columns = 2*2
rows = int(len(d)/2)
l = 0
fig=plt.figure(figsize=(5*columns, 5*rows))
# ax enables access to manipulate each of subplots
ax = []
l = 0

for i in range(1, columns*rows +1,2):
    
    img = plt.imread(d[l])
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title(str(d[l]))
    plt.imshow(img)
    
    ax.append(fig.add_subplot(rows, columns, i+1))
    img = cv2.imread(d[l])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # color = ('b','g','r')
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([img],[i],None,[256],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # # ax[-1].set_title(str(d[l]))
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr,color = 'w')
    plt.xlim([0,256])
    l +=1
plt.show()

#--------------------------HASHING----------------------#


IMGDIR = "./imagesbooks/"

haystackPaths = list(paths.list_images(IMGDIR))
print(haystackPaths)

# init a hash dataframe
haystack = pd.DataFrame(columns=['file', 'phash', 'ahash', 'dhash', 'whash'])

# time the hashing operation 
start = time.time()

for f in haystackPaths:
    
    image = Image.open(f)
#     imageHash = imagehash.phash(image)
    p = imagehash.phash(image, hash_size=8)
    a = imagehash.average_hash(image, hash_size=8)
    d = imagehash.dhash(image, hash_size=8)
    w = imagehash.whash(image, hash_size=8)

    haystack = haystack.append ({'file':f, 'phash':p, 'ahash':a, 'dhash':d,'whash':w }, ignore_index=True)
# print (haystack.head())

#     print (p, imageHash)
    
#     haystack[imageHash] = p

# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")




# -------------------------search images----------------------------# 

import pandas as pd 
import random
# sample = ['./imagesbooks/ukbench00456.jpg']
sample = ['./imagesbooks/ukbench03036.jpg']
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
    p = imagehash.phash(image, hash_size=8)
    a = imagehash.average_hash(image, hash_size=8)
    d = imagehash.dhash(image, hash_size=8)
    w = imagehash.whash(image, hash_size=8)
    
    hashes['phash']= hashes['phash'] - p
    hashes['ahash']= hashes['ahash'] - a
    hashes['dhash']= hashes['dhash'] - d
    hashes['whash']= hashes['whash'] - w
    print(hashes)

    # hashes['phash'].plot()
    # plt.show()    
    # hashes['dhash'].plot()
    # plt.show()    
    # hashes['ahash'].plot()
    # plt.show()    
    # hashes['whash'].plot()
    # plt.show()


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

