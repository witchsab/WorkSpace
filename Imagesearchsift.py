#--------------------------------SIFT CODE------------------------------------#

# top = hashes.sort_values(by=['dhash'])[:20]
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
IMGDIR = "./imagesbooks/"

haystackPaths = list(paths.list_images(IMGDIR))
d = haystackPaths


import cv2

def gen_sift_features(image):
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


# init a sift dataframe
siftdf = pd.DataFrame(columns=['file', 'siftkey', 'siftdesc'])

# time the hashing operation 
start = time.time()

# FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)


# Hyper-Parameters for SIFT comparison
sift_features_limit = 200
lowe_ratio = 0.8
predictions_count = 50


for j in d:
    matches_count = 0
    m_path = j
    m_img = cv2.imread(m_path)        
    if m_img is None:
        continue
    m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
    
    # Generating SIFT features for predicted ssim images
    m_kp,m_des = gen_sift_features(m_img)
    siftdf = siftdf.append({'file':j, 'siftkey':m_kp, 'siftdesc':m_des}, ignore_index=True)
    if m_des is None:
        continue



# Reading query image
# q_path = random.sample(haystackPaths, 5)[0]
q_path = './imagesbooks/ukbench05968.jpg'  # sample test 
q_img = cv2.imread(q_path)    
q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
# Generating SIFT features for query image
q_kp,q_des = gen_sift_features(q_img)


start = time.time()

predictions = []
matches_flann = []

for index, j in siftdf.iterrows(): 
    m_des = j['siftdesc'] 
    m_path = j['file']     
    # Calculating number of feature matches using FLANN
    matches = flann.knnMatch(q_des,m_des,k=2)

    #ratio query as per Lowe's paper
    matches_count = 0
    for x,(m,n) in enumerate(matches):
        if m.distance < lowe_ratio*n.distance:
            matches_count += 1
    matches_flann.append((matches_count,m_path))

matches_flann.sort(key=lambda x : x[0] , reverse = True)
predictions.append((q_path,matches_flann[:predictions_count]))
print(predictions)

t= time.time() - start
print("[INFO] processed {} images in {:.2f} seconds".format(
len(haystackPaths), t))
print( 'Image = %s' %q_path,',', 'Time =%f seconds'%t,'kp=%i' %sift_features_limit)




columns = 5
rows = 10
l = 0
size =2 
fig=plt.figure(figsize=(size*columns, size*rows))

# ax enables access to manipulate each of subplots
ax = []

x,mylist = predictions[0]

for i in range(1, columns*rows +1):
    b,a = mylist [l]
    img = plt.imread(a)
    ax.append(fig.add_subplot(rows, columns, i))
    ax[-1].set_title('score='+str(b))
    plt.imshow(img)
    l +=1

fig.suptitle('Add Title Here', fontsize=16)
plt.show()




sift_score = pd.DataFrame (columns=['score'])
for key in mylist: 
    b, a = key 
    sift_score = sift_score.append(
        {
            'score' : b
        }, ignore_index=True
    )

sift_score.plot()
plt.title('kp=' + str(sift_features_limit) + ', T=' + str(t)  )
plt.show()
print( 'Image = %s' %q_path,',', 'Time =%f seconds'%t)
#-----------------accuracy--------------#



