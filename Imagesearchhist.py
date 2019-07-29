#--------RGB histogram match---------------------#

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




IMGDIR = "./imagesbooks/"

# Hyper-Parameter for comparing histograms
correl_threshold = 0.70

trainhistpath = list(paths.list_images(IMGDIR))
# print (TrainhistPaths)

# init RGB dataframe for Training image lib-------#
Trainhist = pd.DataFrame(columns=['file','imagehist'])

start = time.time()

for f in trainhistpath:
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
print("[INFO] processed {} images in {:.2f} seconds".format(
len(trainhistpath), t))
print (Trainhist.head())



#----------query image gen--------------#

img_path = random.sample(trainhistpath, 1)[0]
print (img_path)

# visualizing the histogram
def showHistogram (img_path):
    import matplotlib.pyplot as plt
    img = cv2.imread(img_path)
    color = ('b','g','r')
    plt.figure(figsize=(15,10))
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
showHistogram (img_path)


# reading query image

q_path = random.sample(trainhistpath, 1)[0]
q_img = cv2.imread(q_path)
query_paths = [q_path]

hist_query = []
for path in query_paths:
    image = cv2.imread(path)
    
    if image is None:
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # extract a RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)
    hist_query.append((path,hist))


print (hist_query)
    

# Histmatch = pd.DataFrame(colums= ['file','matches'])
hist_matches = []
for i in range(len(hist_query)):
    matches = []

    for index, row in Trainhist.iterrows():
        
        cmp = cv2.compareHist(hist_query[i][1], row['imagehist'], cv2.HISTCMP_CORREL)
        
        if cmp > correl_threshold:
            matches.append((cmp, row['file']))

    matches.sort(key=lambda x : x[0] , reverse = True)
    hist_matches.append((hist_query[i][0],matches))

print (hist_matches)


# DATA 

d = [] 
p = []

r , q = hist_matches[0]
for item in q: 
    score , thispath = item 
    d.append(str(thispath))
    p.append(score)

simg = plt.imread(r)
plt.imshow(simg)
plt.show()
# d => list of image paths 
# d = 

# p => list of corresp scores 

# PLOTTING FUNCTION 

fig=plt.figure(figsize=(40, 40))
columns = len(d)
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

fig=plt.figure(figsize=(20, 20))
# ax enables access to manipulate each of subplots
ax = []
l = 0
for i in range(1, columns*rows +1):
    img = cv2.imread(d[l])
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title('score='+str(p[l]))
    l +=1
plt.show()


showHistogram(r)
showHistogram(d[1])