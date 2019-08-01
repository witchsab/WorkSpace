

#--------RGB functional form---------------------#

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



#-----------Training Images RGB Hist GENERRATION----------#
def HSV_GEN(custompaths):
    # init RGB dataframe for Training image lib-------#
    Trainhist = pd.DataFrame(columns=['file','imagehist'])

    start = time.time()

    for f in custompaths:
        image = cv2.imread(f)
        if image is None:
            continue
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # extract a RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 12, 3],[0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None)
        Trainhist =  Trainhist.append({'file':f,'imagehist':hist}, ignore_index=True)

    t= time.time() - start
    # print("[INFO] processed {} images in {:.2f} seconds".format(len(custompaths), t))
    # print (Trainhist.head())
    return (Trainhist,t)


#------------------------------------END--------------------------------------#



#----------query image gen--------------#



def HSV_SEARCH(feature, searchimagepath, correl_threshold):
    start = time.time()
    image = cv2.imread(searchimagepath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # extract a RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 12, 3],[0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)       

    matches = []

    for index, row in feature.iterrows():
        
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