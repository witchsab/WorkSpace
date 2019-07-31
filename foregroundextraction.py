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
import cv2 as cv2
from matplotlib import pyplot as plt

# # filename = (r"V:\\Download\\imagesbooks\\ukbench05777.jpg") 
# filename= ('./imagesbooks/ukbench09364.jpg'  )  
# # ukbench02719 , ukbench02722 , ukbench03045 , ukbench02748, ukbench00032, ukbench00126, ukbench00003


def foregroundExtract(filename) :

    img = cv2.imread(filename)
    image = img.copy()
    

    # define the Region of Interest (ROI) 
    # as the coordinates of the rectangle 
    # where the values are entered as 
    # (startingPoint_x, startingPoint_y, width, height) 
    # these coordinates are according to
    # rect = (50,50,450,290)

    Y, X, Z  = img.shape 
    spread = int (min([X,Y])/1.2)
    rect =  ( int( X/2 - spread /2) , int (Y/2 - spread/2) , spread , spread )

    # print (X,Y)
    # print (rect)

    try : 
        # plt.imshow(img), plt.show()
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        # in case this fails
        cv2.grabCut(img.copy(),mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        image = image*mask2[:,:,np.newaxis]

        # transparency conversion 
        b_channel, g_channel, r_channel = cv2.split(img) # or image
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, mask2*255))
    
    except Exception as e: 
        print ("Some Exception")
        img_BGRA = img.copy()
    
    # plt.imshow(img),plt.colorbar(),plt.show()  # show masked image 
    # plt.imshow(mask2),plt.colorbar(),plt.show() # show mask 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_BGRA = cv2.cvtColor(img_BGRA, cv2.COLOR_BGRA2RGBA)

    return image, img_BGRA


# img = foregroundExtract('./imagesbooks/ukbench09364.jpg')
# img2 = cv2.imread('./imagesbooks/ukbench09364.jpg')


# DOC------- CV2 to PIL conversion 

# img = cv2.imread("path/to/img.png")

# # You may need to convert the color.
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# im_pil = Image.fromarray(img)

# # For reversing the operation:
# im_np = np.asarray(im_pil)


def foregroundExtractAndWarp(filename) :

    img = cv2.imread(filename)
    image = img.copy()
    orig = img.copy()
  

    # define the Region of Interest (ROI) 
    # as the coordinates of the rectangle 
    # where the values are entered as 
    # (startingPoint_x, startingPoint_y, width, height) 
    # these coordinates are according to
    # rect = (50,50,450,290)

    Y, X, Z  = img.shape 
    spread = int (min([X,Y])/1.2)
    rect =  ( int( X/2 - spread /2) , int (Y/2 - spread/2) , spread , spread )

    # print (X,Y)
    # print (rect)

    try : 
        # plt.imshow(img), plt.show()
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        # in case this fails
        cv2.grabCut(img.copy(),mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        image = image*mask2[:,:,np.newaxis]

        # transparency conversion 
        b_channel, g_channel, r_channel = cv2.split(img) # or image
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, mask2*255))


        # ----------------from countours draw rotated rectangular bounding box
        contours, _ = cv2.findContours(mask2.copy(), 1, 1) # not copying here will throw an error

        # sort by max area of contour
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][i],reverse=True))
        # show max contour
        maxct= cnts[0]
        rect = cv2.minAreaRect(maxct)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(orig.copy(),[box],0,(0,255,0),2)
        # plt.imshow (img), plt.show()

        # generate the warped image 
        ratio = ratio = orig.shape[0] / 500.0
        warped1 = four_point_transform(orig.copy(), box.reshape(4, 2) * ratio)
        # plt.imshow (warped), plt.show()
        # return warp 

        # Method 2 - 4 point Quad fit inside contour

        # simplify contours and fit a 4 point rect inside it  
        epsilon = 0.1*cv2.arcLength(maxct,True)
        approx = cv2.approxPolyDP(maxct,epsilon,True)
        if len(approx) !=4 :
            approx =  box  # from previous rotatated rectangle bounding box
        img = cv2.drawContours(img, [approx], 0, (255,255,255), 3)
        plt.imshow (img), plt.show()

        # generate the warped image 
        ratio = ratio = orig.shape[0] / 500.0
        warped2 = four_point_transform(orig.copy(), approx.reshape(4, 2) * ratio)
        plt.imshow (warped), plt.show()
        # return warp 


    except Exception as e: 
        print ("Some Exception")
        img_BGRA = img.copy()
        warped1 = img.copy()
        warped2 = img.copy()
    
    # plt.imshow(img),plt.colorbar(),plt.show()  # show masked image 
    # plt.imshow(mask2),plt.colorbar(),plt.show() # show mask 


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_BGRA = cv2.cvtColor(img_BGRA, cv2.COLOR_BGRA2RGBA) # transparency
    warped1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB)
    warped2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)


    return image, img_BGRA, warped1, warped2


# ------------------------ 4 point perspective transform -------------------------

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

 
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

