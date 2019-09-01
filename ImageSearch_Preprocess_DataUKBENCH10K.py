import os
import pickle
import time

import cv2
import numpy as np
from imutils import paths
from PIL import Image, ImageEnhance

import AccuracyGlobal
import foregroundextraction as extract

# # to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2


# --------------------  DEFINE DIRECTORIES ------------------- # 
# for hash all the images in folder / database 

IMGDIR              = r'./ukbench/'
IMGDIRPROCESSED     = ['']*17

IMGDIRPROCESSED[0]  = r'./ukbench_all/imagesbooks1/'
IMGDIRPROCESSED[1]  = r'./ukbench_all/imagesbooks2/'
IMGDIRPROCESSED[2]  = r'./ukbench_all/imagesbooks_FEXT/'         # Warp in foreground BB 1
IMGDIRPROCESSED[3]  = r'./ukbench_all/imagesbooks_FEXTwarp/'     # Warp on foreground BB 2
IMGDIRPROCESSED[4]  = r'./ukbench_all/imagesbooks_F_ALLBB/'      # original img with overlayed BB 
IMGDIRPROCESSED[5]  = r'./ukbench_all/imagesbooks_S320/'         # Resize to w=320 (50%)
IMGDIRPROCESSED[6]  = r'./ukbench_all/imagesbooks_S160/'         # Resize to w=160 (25%)
IMGDIRPROCESSED[7]  = r'./ukbench_all/imagesbooks_EQ2/'           # Equalized Histogram 
IMGDIRPROCESSED[8]  = r'./ukbench_all/imagesbooks_EQRGB/'        # Equalized Histogram 
IMGDIRPROCESSED[9]  = r'./ukbench_all/imagesbooks_CT2.0/'        # increased  contrast
IMGDIRPROCESSED[10]  = r'./ukbench_all/imagesbooks_S32/'         # Resize to w=32 (very small)
IMGDIRPROCESSED[11]  = r'./ukbench_all/imagesbooks_R90/'         # Rotate by 90 deg 
IMGDIRPROCESSED[12]  = r'./ukbench_all/imagesbooks_R180/'         # Rotate by 180 deg 
IMGDIRPROCESSED[13]  = r'./ukbench_all/imagesbooks_R270/'         # Rotate by 270 deg 
IMGDIRPROCESSED[14]  = r'./ukbench_all/imagesbooks_DENOISE1/'         # Denoise  
IMGDIRPROCESSED[15]  = r'./ukbench_all/imagesbooks_DENOISE2/'         # Denoise  
IMGDIRPROCESSED[16]  = r'./ukbench_all/imagesbooks_CTYING/'        # increased  contrast


# check directories, create if not exist
for dir in IMGDIRPROCESSED : 
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# --------------- LOAD SOURCE FILES -------------------#


haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
imagefiles = haystackPaths # [:50]
# print(haystackPaths)


# --------------- STRAT PREPROCESSING -----------------# 

# # ------- RESIZE (50%, 25%)
# start = time.time()
# size = 320, 320 
# for f in imagefiles:
#     # outfile = os.path.splitext(f)[0]
#     outfile = os.path.basename(f).split('.')[0]
#     if f != outfile:
#         try:
#             im = Image.open(f)
#             im.thumbnail(size, Image.ANTIALIAS)
#             im.save(IMGDIRPROCESSED[5] + outfile + '.jpg', "JPEG")
#         except IOError:
#             print ('cannot create thumbnail')

# size = 160, 160 
# for f in imagefiles:
#     # outfile = os.path.splitext(f)[0]
#     outfile = os.path.basename(f).split('.')[0]
#     if f != outfile:
#         try:
#             im = Image.open(f)
#             im.thumbnail(size, Image.ANTIALIAS)
#             im.save(IMGDIRPROCESSED[6] + outfile + '.jpg', "JPEG")
#         except IOError:
#             print ('cannot create thumbnail')

# size = 32, 32
# for f in imagefiles:
#     # outfile = os.path.splitext(f)[0]
#     outfile = os.path.basename(f).split('.')[0]
#     if f != outfile:
#         try:
#             im = Image.open(f)
#             im.thumbnail(size, Image.ANTIALIAS)
#             im.save(IMGDIRPROCESSED[10] + outfile + '.jpg', "JPEG")
#         except IOError:
#             print ('cannot create thumbnail')

# print("[INFO] Resize (3x) processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))


# ------- REMAP ORIGINAL (GRAYSCALE)

# haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
# imagefiles = haystackPaths # [:50]

# for f in imagefiles:
#     outfile = os.path.basename(f).split('.')[0]
#     # Load the image in greyscale
#     img = cv2.imread(f,0)

#     # create a CLAHE object (Arguments are optional).
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     out = clahe.apply(img)

#     cv2.imwrite (IMGDIRPROCESSED[7] + outfile + '.jpg', out)

#     # # Display the images side by side using cv2.hconcat
#     # out1 = cv2.hconcat([img,out])
#     # cv2.imshow('a',out1)
#     # cv2.waitKey(0)


# ------- REMAP ORIGINAL (COLOR)

haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
imagefiles = haystackPaths # [:50]
print("[INFO] Started Prepocessing for ", len(imagefiles), "images in", IMGDIR)



start = time.time()

for f in imagefiles:
    outfile = os.path.basename(f).split('.')[0]
    img = cv2.imread(f, 1)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    cv2.imwrite (IMGDIRPROCESSED[7] + outfile + '.jpg', img2)

    # # Display the images side by side using cv2.hconcat
    # out1 = cv2.hconcat([img,out])
    # cv2.imshow('a',out1)
    # cv2.waitKey(0)

print("[INFO] Remap COLOR processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))


# # ------- REMAP ORIGINAL (RGB)
# POOR PERFORMANCE 

# haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
# imagefiles = haystackPaths # [:50]
# start = time.time()
# for f in imagefiles:
#     outfile = os.path.basename(f).split('.')[0]

#     # Load image
#     image_bgr = cv2.imread(f)
#     # Convert to YUV
#     image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
#     # Apply histogram equalization
#     image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
#     # Convert to RGB
#     image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

#     cv2.imwrite (IMGDIRPROCESSED[8] + outfile + '.jpg', image_rgb)
# print("[INFO] Remap processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))



# # -------- ENHANCE CONTRAST 

# haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
# imagefiles = haystackPaths # [:50]

# start = time.time()
# for f in imagefiles:
#     # f = haystackPaths[5]
#     im = Image.open(f)
#     enhancer = ImageEnhance.Contrast(im)
#     enhanced_im = enhancer.enhance(2.0) # change to 4.0 or more if shows value 
#     filename = os.path.basename(f).split('.')[0]
#     enhanced_im.save(IMGDIRPROCESSED[9] + filename + '.jpg', format="JPEG")
# print("[INFO] Contrast (2.0) processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))


# # -------- YING = ADAPTIVE CONTRAST, HISTOGRAM EQALIZATION 
# # very very slow
# # haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
# imagefiles = haystackPaths # [:50]
# # import adptiveContrastYing as ying

# # start = time.time()
# # for f in imagefiles:
# #     result = ying.adaptive_ying(f)

# #     filename = os.path.basename(f).split('.')[0]
# #     cv2.imwrite (IMGDIRPROCESSED[16] + filename + '.jpg', result)
    
# # print("[INFO] Contrast (ying) processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))



# # ------- ROTATE ORIGINAL  90, 180, 270  

# haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
# imagefiles = haystackPaths # [:50]
# start = time.time()

# for f in imagefiles:
#     # f = haystackPaths[5]
#     im = Image.open(f)
#     # Rotate it by 45 degrees
#     rotated = im.rotate(90, expand=True)    
#     # save
#     filename = os.path.basename(f).split('.')[0]
#     rotated.save(IMGDIRPROCESSED[11] + filename + '.jpg', format="JPEG")

#     rotated = im.rotate(180, expand=True)    
#     # save
#     filename = os.path.basename(f).split('.')[0]
#     rotated.save(IMGDIRPROCESSED[12] + filename + '.jpg', format="JPEG")

#     rotated = im.rotate(270, expand=True)    
#     # save
#     filename = os.path.basename(f).split('.')[0]
#     rotated.save(IMGDIRPROCESSED[13] + filename + '.jpg', format="JPEG")

# print("[INFO] Rotate (3x) processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))


# # ------- DE-NOISE 

# haystackPaths = sorted(list(paths.list_images(IMGDIR))) #[:2]
# imagefiles = haystackPaths # [:50]
# start = time.time()

# for f in imagefiles:
#     # f = haystackPaths[5]
#     img = cv2.imread(f) 
    
#     # denoising of image saving it into dst image 
#     dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) # changw 21 to 15 or vice versta 
#     # save
#     filename = os.path.basename(f).split('.')[0]
#     cv2.imwrite (IMGDIRPROCESSED[15] + filename + '.jpg', dst)

# print("[INFO] DENOISE processed {} images in {:.2f} seconds".format(len(haystackPaths), time.time() - start))


# ------- ADD NOISE 
# Not Implemented 



# ------- UPDATE ACCURACY DICT in each directory 
# dict source 
# dict 


# accuracy = AccuracyGlobal.AccuracyGlobal() # empty class genrated 

# def update_Preprocessed_Dicts() :
#     IMGDIRPROCESSED.append(IMGDIR) 
#     for DIR in IMGDIRPROCESSED :
#         # DIR = IMGDIRPROCESSED[5] # test 
#         thisDict = {}
#         haystackPaths = sorted(list(paths.list_images(DIR))) #[:2]
#         imagefiles = haystackPaths # [:50]
#         # print(haystackPaths)

#         for f in imagefiles: 
#             gkey, gList = accuracy.accuracy_groundtruth_gen(f)
#             thisDict[gkey] = gList
#         # print (gList)
#         # store the file list 
#         seedFile = DIR + 'seed'
#         outfile = open (seedFile + '.pickle', 'wb')
#         pickle.dump( thisDict.keys, outfile )
#         print ("[INFO] Saving file ", seedFile)

#         # store the file matches (ground truth) dictionary 
#         matchesFile = DIR + 'groundTruth'
#         outfile = open (matchesFile + '.pickle', 'wb')
#         pickle.dump( thisDict, outfile )
#         print ("[INFO] Saving file ", matchesFile)

# # Run an updates on the preprocessed dicts 
# update_Preprocessed_Dicts()



# # ######### Data519 MIX - Org, Rot 3x => Total 520x4 = 2080. update dict  ##########

# def create_augment(): 
#     AUGMENT_DIR = './images/imagesbooks_AUG/'
#     if not os.path.exists(os.path.dirname(AUGMENT_DIR)):
#             try:
#                 os.makedirs(os.path.dirname(AUGMENT_DIR))
#             except OSError as exc: # Guard against race condition
#                 if exc.errno != errno.EEXIST:
#                     raise

#     AUGMENT_LIST = [
#         ('./imagesbooks/', ''),
#         ('./images/imagesbooks_R90/', '_R90'),
#         ('./images/imagesbooks_R180/', '_R180'),
#         ('./images/imagesbooks_R270/', '_R270')]

#     # Consolidate Copy to AUG directory 
#     from shutil import copyfile
#     accuracy = AccuracyGlobal.AccuracyGlobal() # empty class genrated 

#     for DIR, NAME in AUGMENT_LIST:
#         haystackPaths = sorted(list(paths.list_images(DIR))) #[:2]
#         for srcPath in haystackPaths: 
#             src = srcPath
#             fn = (os.path.basename(src)).split('.')[0]+ NAME
#             fe = (os.path.basename(src)).split('.')[1]
#             fname = fn +'.' + fe
#             des = os.path.join(AUGMENT_DIR, fname )
#             # adding exception handling
#             try:
#                 copyfile(src, des)
#                 print (src, des)
#             except IOError as e:
#                 print("Unable to copy file. %s" % e)
#             except:
#                 print("Unexpected error:", sys.exc_info())

#     # generate GroundTruth dict file 

#     KEYDIR, _ = AUGMENT_LIST[0]
#     haystackPaths = sorted(list(paths.list_images(KEYDIR))) #[:2]
#     imagefiles = haystackPaths # [:50]

#     globalDict = {}
#     for f in imagefiles: 
#         gkey, gList = accuracy.accuracy_groundtruth_gen(f)

#         gkeys = []
#         gLists = []

#         for item, name in AUGMENT_LIST : 
#             fn = (gkey).split('.')[0]+ name
#             fe = (gkey).split('.')[1]
#             fname = fn +'.' + fe
#             gkeys.append (fname)
        
#         for gitem in gList : 
#             for item, name in AUGMENT_LIST : 
#                 fn = (gitem).split('.')[0]+ name
#                 fe = (gitem).split('.')[1]
#                 fname = fn +'.' + fe
#                 gLists.append (fname)
        
#         # print ('gkeys', gkeys)
#         # print ('glists', gLists)

#         # Update the global dict globalDict
#         for item in gkeys: 
#             globalDict[item] = gLists

#     # print (gList)
#     # store the file matches (ground truth) dictionary 
#     matchesFile = AUGMENT_DIR + 'groundTruth'
#     outfile = open (matchesFile + '.pickle', 'wb')
#     pickle.dump( globalDict, outfile )
#     print ("[INFO] Saving file ", matchesFile)

#     # store the file list 
#     seedFile = AUGMENT_DIR + 'seed'
#     outfiles = open (seedFile + '.pickle', 'wb')
#     pickle.dump( globalDict.keys, outfiles )
#     print ("[INFO] Saving file ", seedFile)


#     # # check gt file 
#     # accuracy = AccuracyGlobal.AccuracyGlobal() # empty class genrated 
#     # accuracy.read(AUGMENT_DIR)
#     # gt = accuracy.check_ground_truth()


# # create_augment()


# ###########################   DONE   ##############################



# # ------- FOREGROUND EXTRACT & WARP
# def run_foregroundExtractor (): 
#     # time the hashing operation 
#     start = time.time()

#     # image1, image2, image3, image4, bbmap = extract.foregroundExtractAndWarp(haystackPaths[1])    # check 
#     counter = 1 
#     for f in imagefiles:
#         image_orig = Image.open(f)
#         # image1, image2 = extract.foregroundExtract(f)    # check 
#         image1, image2, image3, image4, bbmap = extract.foregroundExtractAndWarp(f)    # check 

#         img1 = Image.fromarray(image1)          # 0 filled bg                # check 
#         img2 = Image.fromarray(image2)          # transparent bg             # check 
#         img3 = Image.fromarray(image3)          # warped bounding box        # check 
#         img4 = Image.fromarray(image4)          # warped inner bounding box  # check 
#         img5 = Image.fromarray(bbmap)           # warped inner bounding box  # check 
#         filename = os.path.basename(f).split('.')[0]
#         img1.save( IMGDIRPROCESSED[0] + filename + '.png', format='PNG')
#         img2.save( IMGDIRPROCESSED[1] + filename + '.png', format='PNG')
#         img3.save( IMGDIRPROCESSED[2] + filename + '.png', format='PNG')
#         img4.save( IMGDIRPROCESSED[3] + filename + '.png', format='PNG')
#         img5.save( IMGDIRPROCESSED[4] + filename + '.png', format='PNG')
#         print ("Processed " , counter , ' ', filename)
#         counter += 1

#     print("[INFO] processed {} images in {:.2f} seconds".format( len(haystackPaths), time.time() - start))
#     # ------- END 
