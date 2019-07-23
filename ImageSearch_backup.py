import PIL
from PIL import Image
import imagehash
import os
import time
from pprint import pprint
from imutils import paths
import matplotlib.pyplot as plt

print('Hello here')

IMGDIR = "./imagesbooks/"
# IMGDIR = "../../images_holidays/jpg/"
# TEST_IMGDIR = "../../test_images/"

haystackPaths = list(paths.list_images(IMGDIR))
# needlePaths = list(paths.list_images(TEST_IMGDIR))
# print (haystackPaths)s

def get_Hash(image, hash_size=32):
    return imagehash.dhash(image, hash_size=hash_size)

    # init a hash dictionary 
haystack = {}

# time the hashing operation 
start = time.time()

for p in haystackPaths:
    
    image = Image.open(p)
#     imageHash = imagehash.phash(image)
    imageHash = get_Hash(image)
#     print (p, imageHash)
    
    haystack[imageHash] = p

# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# for search images 
needlehash = {} 

import pandas as pd 
import random
sample = random.sample(haystackPaths, 5)
# sample = ['./images/ukbench00019.jpg', './images/ukbench00025.jpg', './images/ukbench00045.jpg', './images/ukbench00003.jpg', './images/ukbench00029.jpg']
# sample = ['./images/ukbench00048.jpg', './images/ukbench00016.jpg', './images/ukbench00045.jpg']

print (sample)

for p in sample:
    matchhash = {}
#     print ("Searching", p)
    image = Image.open(p)
        
#     imgplot = plt.imshow(image)
#     plt.show()
    
    hashes = pd.DataFrame(columns=['file', 'value'])    
    
    imageHash = get_Hash(image)    
#     needlehash[imageHash] = p
    
    for key in haystack.keys():
        h = (key - imageHash)
#         h = distance.hamming(key, imageHash)
#         print (h, haystack[key])        
        # matchhash[key] = h
        hashes = hashes.append ({'file':haystack[key], 'value':h }, ignore_index=True)
#     print (df)
    print(hashes)
    top = hashes.sort_values(by=['value'])[:10]
    d = list(top['file'])
    p = list(top['value'])
    # print (d)




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
    
print ("Done") 




print(hashes)

hashes['value'].plot()
plt.show()
