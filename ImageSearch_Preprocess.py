from imutils import paths
from PIL import Image
import time
import foregroundextraction as extract
import os 

# for hash all the images in folder / database 

IMGDIR = r"V:\\Download\\imagesbooks\\"
IMGDIRPROCESSED1 = r"V:\\Download\\imagesbooks1\\"
IMGDIRPROCESSED2 = r"V:\\Download\\imagesbooks2\\"
IMGDIRPROCESSED3 = r"V:\\Download\\imagesbooks3\\"
IMGDIRPROCESSED4 = r"V:\\Download\\imagesbooks4\\"


haystackPaths = list(paths.list_images(IMGDIR)) #[:2]
# print(haystackPaths)

# time the hashing operation 
start = time.time()

counter = 1 
for f in haystackPaths:
    
    image_orig = Image.open(f)
    # image1, image2 = extract.foregroundExtract(f)    # check 
    image1, image2, image3, image4 = extract.foregroundExtractAndWarp(f)    # check 

    img1 = Image.fromarray(image1)         # 0 filled bg                # check 
    img2 = Image.fromarray(image2)         # transparent bg             # check 
    img3 = Image.fromarray(image3)         # warped bounding box        # check 
    img4 = Image.fromarray(image4)         # warped inner bounding box  # check 
    filename = os.path.basename(f).split('.')[0]
    img1.save( IMGDIRPROCESSED1 + filename + '.png', format='PNG')
    img2.save( IMGDIRPROCESSED2 + filename + '.png', format='PNG')
    img3.save( IMGDIRPROCESSED3 + filename + '.png', format='PNG')
    img4.save( IMGDIRPROCESSED4 + filename + '.png', format='PNG')
    print ("Processed " , counter , ' ', filename)
    counter += 1

print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystackPaths), time.time() - start))