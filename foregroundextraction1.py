# ---------------- FOREGROUND EXTRACTION (GRABCUT ALGORITHM) ----------------
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
'''
img - Input image
mask - It is a mask image where we specify which areas are background, foreground or probable background/foreground etc. It is done by the following flags, cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, or simply pass 0,1,2,3 to image.
rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
bdgModel, fgdModel - These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays of size (1,65).
iterCount - Number of iterations the algorithm should run.
mode - It should be cv2.GC_INIT_WITH_RECT or cv2.GC_INIT_WITH_MASK or combined which decides whether we are drawing rectangle or final touchup strokes.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# filename = (r"V:\\Download\\imagesbooks\\ukbench05777.jpg") 
filename = (r"V:\\Download\\imagesbooks\\ukbench05777.jpg")
# ukbench02719 , ukbench02722 , ukbench03045 , ukbench02748, ukbench00032, ukbench00126, ukbench00003

img = cv2.imread(filename)
plt.imshow(img), plt.show()
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# define the Region of Interest (ROI) 
# as the coordinates of the rectangle 
# where the values are entered as 
# (startingPoint_x, startingPoint_y, width, height) 
# these coordinates are according to
# rect = (50,50,450,290)

Y, X, Z  = img.shape 
spread = int (min([X,Y])/1.2)
rect =  ( int( X/2 - spread /2) , int (Y/2 - spread/2) , spread , spread )

print (X,Y)
print (rect)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()  # show masked image 
plt.imshow(mask2),plt.colorbar(),plt.show() # show mask 


