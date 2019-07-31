
#------------------------------------HASH CODE-----------------------------------#

import PIL
from PIL import Image
import imagehash
import os
import cv2
import time
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd 


def HASH_GEN(haystackPaths , hashsize):
    
    # init a hash dataframe
    haystack = pd.DataFrame(columns=['file', 'phash', 'ahash', 'dhash', 'whash'])

    # time the hashing operation 
    start = time.time()

    for f in haystackPaths:
        
        image = Image.open(f)
    #     imageHash = imagehash.phash(image)
        p = imagehash.phash(image, hash_size=hashsize)
        a = imagehash.average_hash(image, hash_size=hashsize)
        d = imagehash.dhash(image, hash_size=hashsize)
        w = imagehash.whash(image, hash_size=hashsize)

        haystack = haystack.append ({'file':f, 'phash':p, 'ahash':a, 'dhash':d,'whash':w }, ignore_index=True)

    # print (haystack.head())
    #     print (p, imageHash)
        
    #     haystack[imageHash] = p

    # show timing for hashing haystack images, then start computing the
    # hashes for needle images
    print("[INFO] processed {} images in {:.2f} seconds".format(
    len(haystack), time.time() - start))    

    return haystack


def HASH_SEARCH (searchImagePath, features, matchCount, hashAlgo, hashsize) : 

    # print (searchImagePath)
       
    # print ("Searching", p)
    image = Image.open(searchImagePath)

    # hashes = pd.DataFrame(columns=['file', 'phash', 'ahash', 'dhash', 'whash'])

    # time the searching operation 
    start = time.time()

    hashes =  features[['file', hashAlgo]].copy()

    # imageHash = imagehash.phash(image)
    

    if hashAlgo == 'phash':
        hashvalue = imagehash.phash(image, hash_size=hashsize)
    elif hashAlgo == 'dhash':
        hashvalue = imagehash.dhash(image, hash_size=hashsize)
    elif hashAlgo == 'ahash':
        hashvalue = imagehash.average_hash(image, hash_size=hashsize)
    elif hashAlgo == 'whash':
        hashvalue = imagehash.whash(image, hash_size=hashsize)
    # a = imagehash.average_hash(image, hash_size=hashsize)
    # d = imagehash.dhash(image, hash_size=hashsize)
    # w = imagehash.whash(image, hash_size=hashsize)    
    
    
    hashes[hashAlgo]= hashes[hashAlgo] - hashvalue
    # hashes['ahash']= hashes['ahash'] - a
    # hashes['dhash']= hashes['dhash'] - d
    # hashes['whash']= hashes['whash'] - w
    # print(hashes)

    ## plot the differences in hash by hash algo type 
    # hashes['phash'].plot()
    # plt.show()    
    # hashes['dhash'].plot()
    # plt.show()    
    # hashes['ahash'].plot()
    # plt.show()    
    # hashes['whash'].plot()
    # plt.show()

    # for item in list(['phash','ahash','dhash','whash']):
    top = hashes.sort_values(by=[hashAlgo])[:matchCount]  # get top 20 matches 
    
    t = time.time() - start
    print("[INFO] processed {} images in {:.2f} seconds".format(
    len(hashes), t))

    # print (top.head())

    # plotting results 
    d = list(top['file'])
    p = list(top[hashAlgo])
    # print (d)
    
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

    # return time and a list of tuples: [( score, file path) ]    
    return (top[[hashAlgo, 'file']].apply(tuple, axis=1), t )


# --------------------------TESTING CODE----------------------------


from imutils import paths

# for hash all the images in folder / database 

IMGDIR = r"V:\\Download\\imagesbooks2\\"
# IMGDIR = "./imagesbooks/"
# IMGDIR = "../../images_holidays/jpg/"
# TEST_IMGDIR = "../../test_images/"

haystackPaths = list(paths.list_images(IMGDIR))

features = HASH_GEN (haystackPaths, 32)


# search images 
 
import random

sample = r'V:\\Download\\imagesbooks2\\ukbench07994.png'
# sample = random.sample(haystackPaths, 1)
# sample = ['./images/ukbench00019.jpg', './images/ukbench00025.jpg', './images/ukbench00045.jpg', './images/ukbench00003.jpg', './images/ukbench00029.jpg']
# sample = ['./images/ukbench00048.jpg', './images/ukbench00016.jpg', './images/ukbench00045.jpg']

mydata, mytime = HASH_SEARCH (sample, features, 20, 'phash', 32)
mydata, mytime = HASH_SEARCH (sample, features, 20, 'dhash', 32)
mydata, mytime = HASH_SEARCH (sample, features, 20, 'ahash', 32)
mydata, mytime = HASH_SEARCH (sample, features, 20, 'whash', 32)


import ImageSearch_Plots as myplots
myplots.plot_predictions(mydata, sample)


# -------------------------END TESTING----------------------------
