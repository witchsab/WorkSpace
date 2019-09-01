import os
from io import BytesIO 
from PIL import Image
import random


# Put the 'full' path (IMGDIR) to the local image repository below 
# and defiene Image Type (IMGTYPE) [png/jog]

# IMGDIR = r'H:\WorkSpace_Python\krazybooks\codes\ImageSearchServer\static\images'
# IMGTYPE = '.jpg'

# IMGDIR = r'V:\Download\image_datasets\vacation_dataset'
# IMGTYPE = r'.png'
# IMGDIR = r'/home/towshif/code/python/pyImageSearch/imagesbooks'

IMGDIR = os.path.join(os.getcwd(), 'imagesbooks')
# IMGDIR = os.path.join(os.getcwd(), 'ukbench_all/imagesbooks_EQ2')
# IMGDIR = os.path.join(os.getcwd(), 'ukbench')
IMGTYPE = r'.jpg'


FULL_ID_LIST = [w.replace(IMGTYPE, '') for w in os.listdir(IMGDIR) ]

def get_image_by_id_from_local(pid):
    imagelink = os.path.join(IMGDIR, pid + IMGTYPE)
    # print (imagelink)
    try: 
        img = Image.open( imagelink, mode='r')
        # resize to 100 px in X or Y whichever is higher 
        img.thumbnail( (100,100), Image.ANTIALIAS)
    except : 
        img = 0
    return img



# Sample list of images from database 
def image_sample_ID(count=3): 
    imgs = []

    # FULL_ID_LIST = [1,2,3,4,5,6,7]  # make database call for final list 
    
    image_ID_list = random.sample( FULL_ID_LIST, count)

    # generate a dictionary {link, text}
    for item in image_ID_list: 
        imgs.append({'ID': item, 'text': 'ID:'+ str(item) })
            
    return imgs