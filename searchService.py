from imutils import paths
import os

# Our own modules
import algos


# ---------------- Main Algo Caller function  ------------------ #

# return lists all image ID in the /static/images directory  
# format: [{'ID': '/imageID', 'text':'hello'}, {...}, {...}]
def imageIDList(input_image_id): 
    imgs = []

    # # Example output: Sample list of tuples from algo module 
    # image_ID_list = [ (1, 20),(2, 5), (3, 7) , (4, 100), (5, 5.89) ,(6, 0.009), (7,66)]

    # call algo1 code : returns a list of tuples 
    image_ID_list = algos.algo1(input_image_id)

    # generate a list of dictionary {ID, text}
    for item in image_ID_list: 
        # Destructure tuple 
        id_value, score_value = item        
        imgs.append({'ID': str(id_value), 'text':'Score='+ str(score_value) })

    return imgs


# ---------------- SOME TESTING FUNCTION ------------------ #

# function: return lists of dicts of all images in the /static/images directory  
# format: [{'link': '/imagelink', 'text':'hello'}, {...}, {...}]
def imagelist(): 
    img = []

    IMGDIR = 'static/images/'
    # print ('PATH', os.getcwd())

    imagelibrary = list(paths.list_images(IMGDIR))
    # print (imagelibrary)
    # generate a dictionary {link, text}
    for item in imagelibrary: 
        i = item.replace (os.getcwd(), '')
        i = '/' + i
        img.append({'link': i, 'text':'Score=0'})

    # print ('list images', img)
    return img