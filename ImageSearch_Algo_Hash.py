
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
import pickle
from sklearn.neighbors import KDTree
import imagehash
import numpy as np
import time


def HASH_GEN ( haystackPaths , hashsize):
    
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

    t = time.time() - start

    print("[INFO] processed {} images in {:.2f} seconds".format(
    len(haystack), t ))    

    return (haystack, t)


def HASH_FEATURE (searchimagepath, hashAlgo='phash', hashsize=8) : 

    queryImage = Image.open(searchimagepath)
    
    if hashAlgo == 'phash':
        hashvalue = imagehash.phash(queryImage, hash_size=hashsize)
    elif hashAlgo == 'dhash':
        hashvalue = imagehash.dhash(queryImage, hash_size=hashsize)
    elif hashAlgo == 'ahash':
        hashvalue = imagehash.average_hash(queryImage, hash_size=hashsize)
    elif hashAlgo == 'whash':
        hashvalue = imagehash.whash(queryImage, hash_size=hashsize)

    return hashvalue


'''
Save Pandas dataframe to pickle 
Datafram format : file , imagehist
'''
def HASH_SAVE_FEATURES ( mydataHASH, savefile='testHASH_Data') : 
    
    # save the tree #example # treeName = 'testRGB_Data.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( mydataHASH, outfile)


    
'''
Load Pandas dataframe from pickle 
Datafram format : file , imagehist
'''
def HASH_LOAD_FEATURES (openfile='testHASH_Data') : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    mydataHASH = pickle.load(infile)
    infile.close()

    return mydataHASH


'''
Create a KDTree with the hash 
params: 
mydataHash  = pandas dataframe; format: 'file', 'phash', 'dhash', 'ahash', 'whash'
savefile    = filename to save pickle (dont add .pickle)
hashAlgo    = phash, dhash, ahash, whash 

output/return: 
HashTree (KDTree)

'''
def HASH_Create_Tree ( mydataHASH, savefile='testHash', hashAlgo='dhash'): 
    
    # YD = np.array(mydataHASH['phash'].apply(imagehash.ImageHash.__hash__))
    YD = list(mydataHASH[hashAlgo])

    # a = np.empty((h, w)) # create an empty array
    result_array = []

    for item in YD : 
        onearray = np.asarray(np.array (item.hash), dtype=float)
        result_array.append(onearray)

    YA = np.asarray(result_array)
    nsamples, x, y = YA.shape  # know the shape before you flatten
    F = YA.reshape ( nsamples, x*y ) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    HASHTree = KDTree(F, metric='euclidean')
    
    # save the tree #example # treeName = 'testHash.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump(HASHTree,outfile)

    return HASHTree



def HASH_Load_Tree ( openfile='testHash'  ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    HashTree = pickle.load(infile)
    infile.close()

    return HashTree


'''
Params: 
HSVTree = Tree object 
mydataHSV = pandas dataframe of the tree (same order no filter or change)
searchimagepath = string path of the search image 

Output: 
list of tuples: [(score, matchedfilepath) ]
time = total searching time 
'''
def HASH_SEARCH_TREE ( HASHTree , mydataHASH,  searchimagepath, hashAlgo = 'dhash', hashsize=8, returnCount=100): 

    start = time.time()
    # convert to np array from ImageHash->hash  
    fh = np.array(HASH_FEATURE(searchimagepath, hashAlgo=hashAlgo, hashsize=hashsize).hash)

    fd = np.asarray(fh , dtype=float) # convert to numpy float array for tree 
    
    # reshape to 1xdim array to feed into tree
    x, y = fd.shape # know the shape before you flatten
    FF = fd.reshape (1, x*y) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    scores, ind = HASHTree.query(FF, k=returnCount)
    t = time.time() - start 

    # Zip results into a list of tuples (score , file) & calculate score 
    flist = list (mydataHASH.iloc[ ind[0].tolist()]['file'])
    slist = list (scores[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists

    return (matches, t)



def HASH_CREATE_HYBRIDTREE ( mydataHASH, savefile='testHash', hashAlgoList=['whash', 'ahash'] ) :  
    # a = np.empty((h, w)) # create an empty array 
    result_array = []

    for index, row in mydataHASH.iterrows() :      
        thisarray = []
        for algo in hashAlgoList : 
            hashValue =  row[algo].hash        
            thisarray.append(np.asarray(np.array (hashValue), dtype=float))
        
        result_array.append (np.asarray(thisarray, dtype=float))

    YA = np.asarray(result_array)

    nsamples, x, y, z = YA.shape  # know the shape before you flatten
    F = YA.reshape ( nsamples, x*y*z ) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    HybridHASHTree = KDTree( F ,  metric='euclidean')

    # save the tree #example # treeName = 'testHash.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump(HybridHASHTree,outfile)

    return HybridHASHTree


def HASH_SEARCH_HYBRIDTREE ( HybridHASHTree , mydataHASH,  searchimagepath, hashAlgoList = [ 'whash', 'ahash'], hashsize=8, returnCount=100): 

    start = time.time()
    thisarray = []
    for algo in hashAlgoList : 
        hashValue =  HASH_FEATURE( searchimagepath , algo, 16).hash
        thisarray.append(np.asarray(np.array (hashValue), dtype=float))

    # result_array.append (np.asarray(thisarray, dtype=float))

    fd = np.asarray( thisarray , dtype=float) # convert to float array
    # ft = raw feature 
    # process 
    x, y, z = fd.shape # know the shape before you flatten
    FF = fd.reshape (1, x*y*z) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    scores, ind = HybridHASHTree.query(FF, k=returnCount)
    t = time.time() - start 

    # Zip results into a list of tuples (score , file) & calculate score 
    flist = list (mydataHASH.iloc[ ind[0].tolist()]['file'])
    slist = list (scores[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 

    return (matches, t)


def HASH_SEARCH (searchImagePath, features, matchCount=20, hashAlgo='phash', hashsize=8) : 

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
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(hashes), t))

    # print (top.head())

    # # plotting results 
    # d = list(top['file'])
    # p = list(top[hashAlgo])
    # # print (d)
    
    # fig=plt.figure(figsize=(40, 40))
    # columns = 20
    # rows = 1
    # l = 0
    # # ax enables access to manipulate each of subplots
    # ax = []

    # for i in range(1, columns*rows +1):
    #     img = plt.imread(d[l])
    #     ax.append(fig.add_subplot(rows, columns, i))
    #     ax[-1].set_title('score='+str(p[l]))
    #     plt.imshow(img)
    #     l +=1
    # plt.show()

    # return time and a list of tuples: [( score, file path) ]    
    return ( list(top[[hashAlgo, 'file']].apply(tuple, axis=1)), t )


# # --------------------------TESTING CODE----------------------------


# from imutils import paths

# # for hash all the images in folder / database 

# IMGDIR = r"V:\\Download\\imagesbooks2\\"
# # IMGDIR = "./imagesbooks/"
# # IMGDIR = "../../images_holidays/jpg/"
# # TEST_IMGDIR = "../../test_images/"

# haystackPaths = list(paths.list_images(IMGDIR))

# features = HASH_GEN (haystackPaths, 32)


# # search images 
 
# import random

# sample = r'V:\\Download\\imagesbooks2\\ukbench07994.png'
# # sample = random.sample(haystackPaths, 1)
# # sample = ['./images/ukbench00019.jpg', './images/ukbench00025.jpg', './images/ukbench00045.jpg', './images/ukbench00003.jpg', './images/ukbench00029.jpg']
# # sample = ['./images/ukbench00048.jpg', './images/ukbench00016.jpg', './images/ukbench00045.jpg']

# mydata, mytime = HASH_SEARCH (sample, features, 20, 'phash', 32)
# mydata, mytime = HASH_SEARCH (sample, features, 20, 'dhash', 32)
# mydata, mytime = HASH_SEARCH (sample, features, 20, 'ahash', 32)
# mydata, mytime = HASH_SEARCH (sample, features, 20, 'whash', 32)


# import ImageSearch_Plots as myplots
# myplots.plot_predictions(mydata, sample)


# # -------------------------END TESTING----------------------------
