
<!-- FROM dataAnalysis.py -->

# ------------SIFT GENERATION TEST-------------------#

import pandas as pd
import ImageSearch_Algo_SIFT
from imutils import paths

# Hyper-Parameters for SIFT comparison
sift_features_limit = 100
lowe_ratio = 0.75
predictions_count = 50

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
mydataSIFT, mytime1 = ImageSearch_Algo_SIFT.gen_sift_features(imagepaths, sift_features_limit)
print ("SIFT Feature Generation time :", mytime1)

# save to file 
hdfSIFT = pd.HDFStore('SIFT_Features.h5')
hdfSIFT.put('mydataSIFT', mydataSIFT[['file', 'siftdes']], data_columns=True)


<!-- imageSearch_Algo_SIFT.py -->


   
def FEATURE (queryimagepath, sift_features_limit): 
    q_img = cv2.imread(queryimagepath)    
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    sift = cv2.xfeatures2d.SIFT_create(sift_features_limit)
    q_kp, q_des = sift.detectAndCompute(q_img, None)

    return q_des



<!-- KNN Book SIFT functions -->




# --------------------  KD Tree  SIFT ---------------------
# Avg. Time per search: 0.033 s


from sklearn.neighbors import KDTree
import time 

YD = list( mydataSIFT['siftdes'])
YA = np.asarray(YD).flatten()
# nsamples, nx, ny, nz = XA.shape  # know the shape before you flatten
# X = XA.reshape ((nsamples, nx*ny*nz)) # gives a 2 D matice (sample, value) which can be fed to KMeans 

SIDTtree = KDTree(YA) # , metric='euclidean')

#  Example with HSV metrices 
import ImageSearch_Algo_SIFT 
import Accuracy as accuracy

# random sample 
q_path = random.sample(imagepaths, 1)[0]

start = time.time()
# test 
fh = ImageSearch_Algo_HSV.HSV_FEATURE (q_path)

fh = np.asarray(fh)
# ft = raw feature 
# process 
nz = fh.shape  # know the shape before you flatten
F = fh.reshape (1, -1) # gives a 2 D matice (sample, value) which can be fed to KMeans 

dist, ind = SIDTtree.query(F, k=100)
t = time.time() - start 

print (ind)

# get the index of searchimage 
print (mydataHSV.index[mydataHSV['file'] == q_path])
print (q_path)
print ( "Search took ", t, ' secs')

# Zip results into a list of tuples (score , file) & calculate score 
flist = list (mydataHSV.iloc[ ind[0].tolist()]['file'])
slist = list (dist[0])
result = tuple(zip( slist, flist)) # create a list of tuples from 2 lists 
a , q = accuracy.accuracy_matches(q_path, result, 100)
print ('Accuracy =',  a, '%', '| Quality:', q )
