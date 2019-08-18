


#################################################################################
#############################     MERGE CODE    #################################
#################################################################################



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
import random
import ImageSearch_Algo_HSV 
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import Accuracy as accuracy  
from kneed import KneeLocator

# to reload module: uncomment use the following
%load_ext autoreload
%autoreload 2

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytimeHSV = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytimeHSV)

mydataRGB, mytimeRGB = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print ('RGB Feature Generation time', mytimeRGB)

#Loading kp100 sift features
mydatasift = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES('data/SIFT_features_519set_kp100_pandas')


# to create a new tree from dataframe features 'mydataHSV'
mytreeHSV = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')
mytreeRGB = ImageSearch_Algo_RGB.RGB_Create_Tree (mydataRGB, savefile='RGB_Tree')

# q_paths = random.sample(imagepaths, 5)  # random sample 100 items in list

# q_paths = imagepaths  

# q_paths = ['./imagesbooks/ukbench05960.jpg']  #,'./imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg']
# q_paths = ['./imagesbooks/ukbench05960.jpg', './imagesbooks/ukbench00459.jpg', './imagesbooks/ukbench06010.jpg', './imagesbooks/ukbench06104.jpg', './imagesbooks/ukbench00458.jpg', './imagesbooks/ukbench00248.jpg', './imagesbooks/ukbench06408.jpg', './imagesbooks/ukbench00303.jpg', './imagesbooks/ukbench03124.jpg', './imagesbooks/ukbench05776.jpg', './imagesbooks/ukbench06113.jpg', './imagesbooks/ukbench05964.jpg', './imagesbooks/ukbench10164.jpg', './imagesbooks/ukbench02750.jpg', './imagesbooks/ukbench05951.jpg', './imagesbooks/ukbench05983.jpg', './imagesbooks/ukbench03867.jpg', './imagesbooks/ukbench05883.jpg', './imagesbooks/ukbench06049.jpg', './imagesbooks/ukbench06017.jpg', './imagesbooks/ukbench06150.jpg', './imagesbooks/ukbench06151.jpg', './imagesbooks/ukbench02749.jpg', './imagesbooks/ukbench02721.jpg', './imagesbooks/ukbench05879.jpg', './imagesbooks/ukbench06148.jpg', './imagesbooks/ukbench05880.jpg', './imagesbooks/ukbench05929.jpg', './imagesbooks/ukbench06048.jpg', './imagesbooks/ukbench08544.jpg', './imagesbooks/ukbench03058.jpg', './imagesbooks/ukbench10154.jpg', './imagesbooks/ukbench00000.jpg', './imagesbooks/ukbench05972.jpg', './imagesbooks/ukbench05872.jpg', './imagesbooks/ukbench08542.jpg', './imagesbooks/ukbench06004.jpg', './imagesbooks/ukbench05993.jpg', './imagesbooks/ukbench05988.jpg', './imagesbooks/ukbench00483.jpg', './imagesbooks/ukbench08546.jpg', './imagesbooks/ukbench06539.jpg', './imagesbooks/ukbench02748.jpg', './imagesbooks/ukbench05980.jpg', './imagesbooks/ukbench08001.jpg', './imagesbooks/ukbench03890.jpg', './imagesbooks/ukbench03059.jpg', './imagesbooks/ukbench10081.jpg', './imagesbooks/ukbench06519.jpg', './imagesbooks/ukbench05787.jpg']

# q_paths = ["./imagesbooks/ukbench00196.jpg",  "./imagesbooks/ukbench00199.jpg",  "./imagesbooks/ukbench00296.jpg",  "./imagesbooks/ukbench00298.jpg",  "./imagesbooks/ukbench00299.jpg",  "./imagesbooks/ukbench00300.jpg",  "./imagesbooks/ukbench00302.jpg",  "./imagesbooks/ukbench00303.jpg",  "./imagesbooks/ukbench02730.jpg",  "./imagesbooks/ukbench02740.jpg",  "./imagesbooks/ukbench02743.jpg",  "./imagesbooks/ukbench05608.jpg",  "./imagesbooks/ukbench05932.jpg",  "./imagesbooks/ukbench05933.jpg",  "./imagesbooks/ukbench05934.jpg",  "./imagesbooks/ukbench05935.jpg",  "./imagesbooks/ukbench05952.jpg",  "./imagesbooks/ukbench05953.jpg",  "./imagesbooks/ukbench05954.jpg",  "./imagesbooks/ukbench05955.jpg",  "./imagesbooks/ukbench05956.jpg",  "./imagesbooks/ukbench05957.jpg",  "./imagesbooks/ukbench05958.jpg",  "./imagesbooks/ukbench05959.jpg",  "./imagesbooks/ukbench06148.jpg",  "./imagesbooks/ukbench06149.jpg",  "./imagesbooks/ukbench06150.jpg",  "./imagesbooks/ukbench06151.jpg",  "./imagesbooks/ukbench06558.jpg",  "./imagesbooks/ukbench06559.jpg",  "./imagesbooks/ukbench07285.jpg",  "./imagesbooks/ukbench07588.jpg",  "./imagesbooks/ukbench07589.jpg",  "./imagesbooks/ukbench07590.jpg",  "./imagesbooks/ukbench08540.jpg",  "./imagesbooks/ukbench08542.jpg",  "./imagesbooks/ukbench08592.jpg", "./imagesbooks/ukbench08594.jpg","./imagesbooks/ukbench08595.jpg","./imagesbooks/ukbench08609.jpg","./imagesbooks/ukbench09364.jpg","./imagesbooks/ukbench09365.jpg","./imagesbooks/ukbench09366.jpg","./imagesbooks/ukbench10061.jpg","./imagesbooks/ukbench10065.jpg","./imagesbooks/ukbench10066.jpg","./imagesbooks/ukbench10085.jpg","./imagesbooks/ukbench10087.jpg","./imagesbooks/ukbench10108.jpg","./imagesbooks/ukbench10109.jpg","./imagesbooks/ukbench10110.jpg","./imagesbooks/ukbench10112.jpg","./imagesbooks/ukbench10113.jpg","./imagesbooks/ukbench10114.jpg","./imagesbooks/ukbench10116.jpg","./imagesbooks/ukbench10118.jpg","./imagesbooks/ukbench10119.jpg","./imagesbooks/ukbench10124.jpg","./imagesbooks/ukbench10125.jpg","./imagesbooks/ukbench10126.jpg","./imagesbooks/ukbench10128.jpg","./imagesbooks/ukbench10129.jpg","./imagesbooks/ukbench10130.jpg","./imagesbooks/ukbench10131.jpg","./imagesbooks/ukbench10152.jpg","./imagesbooks/ukbench10153.jpg","./imagesbooks/ukbench10154.jpg","./imagesbooks/ukbench10164.jpg","./imagesbooks/ukbench10165.jpg","./imagesbooks/ukbench10166.jpg","./imagesbooks/ukbench10167.jpg"]

q_paths = ["./imagesbooks/ukbench00196.jpg"]

accStatsmerge = pd.DataFrame(columns=['file'])
matcheshsv = []
matchesrgb = []

for q_path in q_paths: 

    toplist = []

    start = time.time()

    row_dict = {'file':q_path }   
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (mytreeHSV, mydataHSV, q_path, 100)

    x = autothreshold (imagematcheshsv)
    toplist = toplist + x
    # print (x)
    a, d, i_hsv, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
    row_dict['hsv_acc'] = a
    row_dict['hsv_matchindex'] = i_hsv
    row_dict['hsv_Count'] = cnt
    row_dict['hsv_time'] = searchtimehsv
    print ('HSV Accuracy =',  a, '%', '| Quality:', d )
    print ('Count', cnt, ' | position', i_hsv)
  

    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (mytreeRGB, mydataRGB, q_path, 100)
    y= autothreshold (imagematchesrgb)
    toplist = toplist + y
    # print (y)
    a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
    row_dict['rgb_acc'] = a
    row_dict['rgb_matchindex'] = i_rgb
    row_dict['rgb_Count'] = cnt
    row_dict['rgb_time'] = searchtimergb
    print ('RGB Accuracy =',  a, '%', '| Quality:', d )
    print ('Count', cnt, ' | position', i_rgb)


    # generate 200 candidates for SIFT (top 100 from each RGB, HSV etc.)
    filteredsift = filtered_SIFT_Candidates(imagematcheshsv, imagematchesrgb, mydatasift)

    # Run SIFT SEARCH for the 200 candidates 
    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(filteredsift, q_path, 100, 0.6, 50)
    a ,d, i_sift, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    print (q_paths.index(q_path), q_path)
    print ('SIFT Merge Accuracy =',  a, '%', '| Quality:', d )
    print ('Count', cnt, ' | position', i_sift)

    row_dict['siftmerge_acc'] = a
    row_dict['siftmerge_matchindex'] = i_sift
    row_dict['siftmerge_Count'] = cnt
    row_dict['siftmerge_quality'] = d
    row_dict['siftmerge_time'] = searchtimesift
    print ("siftmerge processing time", searchtimesift, 'sec')

    t = time.time() - start
    row_dict['totaltime'] = t
    print ('total processing time = ', t, 'seconds')
    # print( toplist)

    # append top 20 results from SIFT to end of TOPLIST 
    for myitem in imagepredictions[:20]:
        x, y = myitem
        toplist.append(y)

    
    # print( toplist)

    # print sanitize_List(toplist))
    toplist = sanitize_List(toplist)

    a ,d, i_toplist, cnt = accuracy.accuracy_from_list(q_path, toplist, 20 )
    print (q_paths.index(q_path), q_path)
    print ('Final Accuracy =',  a, '%', '| Quality:', d )
    print ('Final Count', cnt, ' | position', i_toplist)

    row_dict['toplistmerge_acc'] = a
    row_dict['toplistmerge_matchindex'] = i_toplist
    row_dict['toplistmerge_Count'] = cnt
    row_dict['toplistmerge_quality'] = d
    # row_dict['toplistmerge_time'] = searchtimetoplist
    # print ("toplistmerge processing time", searchtimetoplist, 'sec')

    print ("-------------------------------------------")

    
    
    
    #Sift code only

    #Loading kp100 sift features
    mydatasift = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES('data/SIFT_features_519set_kp100_pandas')

    imagepredictions , searchtimesift = ImageSearch_Algo_SIFT.SIFT_SEARCH(mydatasift, q_path, 100, 0.6, 50)
    a ,d, i_sift, cnt = accuracy.accuracy_matches(q_path, imagepredictions, 20 )
    print (q_paths.index(q_path), q_path)
    print ('Accuracy =',  a, '%', '| Quality:', d )
    print ('Count', cnt, ' | position', i_sift)

    row_dict['siftalone_acc'] = a
    row_dict['siftalone_matchindex'] = i_sift
    row_dict['siftalone_Count'] = cnt
    # row_dict['siftalonequality'] = d
    row_dict['siftalone_time'] = searchtimesift

    accStatsmerge = accStatsmerge.append( row_dict , ignore_index=True)
# print ('HSV Tree Mean Search time', accStatsmerge['hsv'+'_time'].mean())
# print ("Mean Acc = ", accStatsmerge['hsv' + '_predict20'].mean())

accStatsmerge.to_csv('accStatsmergetestmerge519.csv')










def filtered_SIFT_Candidates(imagematches1, imagematches2, mydataframe):
    # create a mergelist 
    mergelist = set ()

    # searchFile, searchResults = matcheshsv[0]

    for myitem in imagematches1:
        x, y = myitem
        mergelist.add(y)

    # searchFile, searchResults = matchesrgb[0]
    for myitem in imagematches2:
        x, y = myitem
        mergelist.add(y)

    mergelist = list(mergelist)

    # print ("Candidate count ", len(mergelist))
    # row_dict['mergecount'] = len(mergelist)
    

    filteredframe = mydataframe[mydataframe['file'].isin(mergelist)]

    return filteredframe




# merge with other algos after thresholding

def autothreshold (imagematches):
    score = []
    successScore = []
    filelist = []
    # score, file = item
    for item in imagematches:
        x, y = item
        score.append(x)
        filelist.append (y)
    # print(score)
    successPositions =i_rgb
    for i in i_rgb: 
        successScore.append(score[i])
    
    # setting default value  
    myknee = 0
    try: 
        elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
        myknee = elbow.knee
        print ('Detected Elbow cluster value :', myknee)
    except: 
        pass

    qualifiedItems = min (myknee , 6)

    return filelist[:qualifiedItems]


'''
Removes duplicates from list while preserving order 
Fast implementation (check stackoverflow)
'''
def sanitize_List(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
