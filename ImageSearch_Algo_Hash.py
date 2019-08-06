
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



def HASH_Create_Tree ( mydataHash, savefile='testHash', hashAlgo='phash' ) : 
    
    YD = list(mydataHash[ hashAlgo ])
    YA = np.asarray(YD)
    # nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
    # X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    Hashtree = KDTree(YA ) # , metric='euclidean')
    
    # save the tree #example # treeName = 'testHash.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump(Hashtree,outfile)

    return Hashtree



def HASH_Load_Tree ( openfile='testHash'  ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    HSVTree = pickle.load(infile)
    infile.close()

    return HSVTree


'''
Params: 
HSVTree = Tree object 
mydataHSV = pandas dataframe of the tree (same order no filter or change)
searchimagepath = string path of the search image 

Output: 
list of tuples: [(score, matchedfilepath) ]
time = total searching time 
'''
def HASH_SEARCH_TREE ( Hashtree , mydataHash,  searchimagepath, returnCount=100): 

    start = time.time()
    
    # get the feature from the input image 
    fh = HASH_FEATURE (searchimagepath)

    fh = np.asarray(fh)
    # ft = raw feature 
    # process 
    nz = fh.shape  # know the shape before you flatten
    F = fh.reshape (1, -1) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    # the search; k = number of returns expected 
    # returns distances, index of the items in the order of the input tree nparray 
    dist, ind = Hashtree.query(F, k=returnCount)
    t = time.time() - start 

    flist = list (mydataHash.iloc[ ind[0].tolist()]['file'])
    slist = list (dist[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 

    return (matches, t)


def HASH_SEARCH (searchImagePath, features, matchCount=20, hashAlgo='phahs', hashsize=8) : 

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
