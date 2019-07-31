from imutils import paths
from PIL import Image
import time
import foregroundextraction as extract
import os 


# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2


# for hash all the images in folder / database 

IMGDIR = r"V:\\Download\\imagesbooks\\"
IMGDIRPROCESSED = ['']*5
IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"


# check directories, create if not exist
for dir in IMGDIRPROCESSED : 
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


haystackPaths = list(paths.list_images(IMGDIR)) #[:2]
# print(haystackPaths)

# time the hashing operation 
start = time.time()


image1, image2, image3, image4, bbmap = extract.foregroundExtractAndWarp(haystackPaths[1])    # check 


counter = 1 
for f in haystackPaths:
    
    image_orig = Image.open(f)
    # image1, image2 = extract.foregroundExtract(f)    # check 
    image1, image2, image3, image4, bbmap = extract.foregroundExtractAndWarp(f)    # check 

    img1 = Image.fromarray(image1)          # 0 filled bg                # check 
    img2 = Image.fromarray(image2)          # transparent bg             # check 
    img3 = Image.fromarray(image3)          # warped bounding box        # check 
    img4 = Image.fromarray(image4)          # warped inner bounding box  # check 
    img5 = Image.fromarray(bbmap)           # warped inner bounding box  # check 
    filename = os.path.basename(f).split('.')[0]
    img1.save( IMGDIRPROCESSED[0] + filename + '.png', format='PNG')
    img2.save( IMGDIRPROCESSED[1] + filename + '.png', format='PNG')
    img3.save( IMGDIRPROCESSED[2] + filename + '.png', format='PNG')
    img4.save( IMGDIRPROCESSED[3] + filename + '.png', format='PNG')
    img5.save( IMGDIRPROCESSED[4] + filename + '.png', format='PNG')
    print ("Processed " , counter , ' ', filename)
    counter += 1

print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystackPaths), time.time() - start))