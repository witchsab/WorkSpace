

#--------RGB functional form---------------------#

import PIL
from PIL import Image
import imagehash
import os
import cv2
import time
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import random
from sklearn.neighbors import KDTree
import os 
import pickle


#-----------Training Images RGB Hist GENERRATION----------#
def RGB_GEN(custompaths):
    # init RGB dataframe for Training image lib-------#
    Trainhist = pd.DataFrame(columns=['file','imagehist'])

    start = time.time()

    for f in custompaths:
        image = cv2.imread(f)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # extract a RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None)
        Trainhist =  Trainhist.append({'file':f,'imagehist':hist}, ignore_index=True)

    t= time.time() - start
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(custompaths), t))
    # print (Trainhist.head())
    return (Trainhist,t)

'''
Save Pandas dataframe to pickle 
Datafram format : file , imagehist
'''
def RGB_SAVE_FEATURES ( mydataRGB, savefile='testRGB_Data') : 
    
    # save the tree #example # treeName = 'testRGB_Data.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump( mydataRGB, outfile)

'''
Load Pandas dataframe from pickle 
Datafram format : file , imagehist
'''
def RGB_LOAD_FEATURES ( openfile='testRGB_Data') : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    mydataRGB = pickle.load(infile)
    infile.close()

    return mydataRGB

#------------------------------------END--------------------------------------#



#----------query image gen--------------#


def RGB_FEATURE (searchimagepath) : 
    image = cv2.imread(searchimagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)

    return hist

def RGB_Create_Tree ( mydataRGB, savefile='testRGB'  ) : 
    
    # for feature vector matrices from features daataframe 
    XD = list(mydataRGB['imagehist'])
    XA = np.asarray(XD)
    nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
    X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    RGBtree = KDTree(X , metric='manhattan')
    
    # save the tree #example # treeName = 'testRGB.pickle'
    outfile = open (savefile + '.pickle', 'wb')
    pickle.dump(RGBtree,outfile)

    return RGBtree



def RGB_Load_Tree ( openfile='testRGB'  ) : 
    
    # reading the pickle tree
    infile = open(openfile + '.pickle','rb')
    RGBTree = pickle.load(infile)
    infile.close()

    return RGBTree


'''
Params: 
HSVTree = Tree object 
mydataHSV = pandas dataframe of the tree (same order no filter or change)
searchimagepath = string path of the search image 

Output: 
list of tuples: [(score, matchedfilepath) ]
time = total searching time 
'''
def RGB_SEARCH_TREE ( RGBtree , mydataRGB,  searchimagepath, returnCount=100): 

    start = time.time()
    
    # get the feature from the input image 
    fh = RGB_FEATURE (searchimagepath)

    fh = np.asarray(fh)
    # ft = raw feature 
    # process 
    # nz = fh.shape  # know the shape before you flatten
    F = fh.reshape (1, -1) # gives a 2 D matice (sample, value) which can be fed to KMeans 

    # the search; k = number of returns expected 
    # returns distances, index of the items in the order of the input tree nparray 
    dist, ind = RGBtree.query(F, k=returnCount)
    t = time.time() - start 

    flist = list (mydataRGB.iloc[ ind[0].tolist()]['file'])
    slist = list (dist[0])
    matches = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 

    return (matches, t)



def RGB_SEARCH(feature, searchimagepath, correl_threshold):
    start = time.time()
    image = cv2.imread(searchimagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)       

    matches = []
    for _ , row in feature.iterrows():        
        cmp = cv2.compareHist(hist, row['imagehist'], cv2.HISTCMP_CORREL)        
        if cmp > correl_threshold:
            matches.append((cmp, row['file']))

    matches.sort(key=lambda x : x[0] , reverse = True)
    t= time.time() - start
    return (matches, t)
#------------------------------END-----------------------------------------#






# # ------------- GENERATION TEST-------------------#

# # Hyper-Parameter for comparing histograms
# parametercorrelationthreshold = 0.70

# IMGDIR = "./imagesbooks/"
# imagepaths = list(paths.list_images(IMGDIR))
# # print (imagepathss)

# mydata, mytime = RGB_GEN(imagepaths)




# #------SEARCH TEST------------------------------#

# q_path = random.sample(imagepaths, 1)[0]
# imagematches , searchtime = RGB_SEARCH(mydata, q_path, 0.7)

# # to reload module: uncomment use the following 
# # %load_ext autoreload
# # %autoreload 2

# import Accuracy as accuracy
# a = accuracy.accuracy_matches(q_path, imagematches, 50)
# print ('Accuracy =',  a, '%')

# import ImageSearch_Plots as myplots
# myplots.plot_predictions(imagematches, q_path)


# #---------------- Compile data and plot results 

# accStats = pd.DataFrame(columns=['file','Acc', 'PCount'])

# q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 

# for q_path in q_paths:    
#     imagematches , searchtime = RGB_SEARCH(mydata, q_path, 0.7)
#     a = accuracy.accuracy_matches(q_path, imagematches, 50)
#     accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': len(imagematches) } , ignore_index=True)


# plt.plot(accStats['Acc'])
# plt.plot (accStats['PCount'])
# plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

# print ("Mean Acc = ", accStats['Acc'].mean())