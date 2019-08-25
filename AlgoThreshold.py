

##############################################################################
#######################------  RGB THRESHOLDING------ ########################
##############################################################################


import ImageSearch_Algo_RGB
from imutils import paths

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = qpaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydata, mytime = ImageSearch_Algo_RGB.RGB_GEN(imagepaths)
print (mytime)



#------SEARCH TEST------------------------------#

# q_path = random.sample(imagepaths, 1)[0]
q_paths = random.sample(imagepaths, 20)  # random sample 100 items in list

q_paths = [
    "./imagesbooks/ukbench00196.jpg", "./imagesbooks/ukbench00199.jpg",  "./imagesbooks/ukbench00296.jpg",  "./imagesbooks/ukbench00298.jpg",  "./imagesbooks/ukbench00299.jpg",  "./imagesbooks/ukbench00300.jpg",  "./imagesbooks/ukbench00302.jpg",  "./imagesbooks/ukbench00303.jpg",  "./imagesbooks/ukbench02730.jpg",  "./imagesbooks/ukbench02740.jpg",  "./imagesbooks/ukbench02743.jpg",  "./imagesbooks/ukbench05608.jpg",  "./imagesbooks/ukbench05932.jpg",  "./imagesbooks/ukbench05933.jpg",  "./imagesbooks/ukbench05934.jpg",  "./imagesbooks/ukbench05935.jpg",  "./imagesbooks/ukbench05952.jpg",  "./imagesbooks/ukbench05953.jpg",  "./imagesbooks/ukbench05954.jpg",  "./imagesbooks/ukbench05955.jpg",  "./imagesbooks/ukbench05956.jpg",  "./imagesbooks/ukbench05957.jpg",  "./imagesbooks/ukbench05958.jpg",  "./imagesbooks/ukbench05959.jpg",  "./imagesbooks/ukbench06148.jpg",  "./imagesbooks/ukbench06149.jpg",  "./imagesbooks/ukbench06150.jpg",  "./imagesbooks/ukbench06151.jpg",  "./imagesbooks/ukbench06558.jpg",  "./imagesbooks/ukbench06559.jpg",  "./imagesbooks/ukbench07285.jpg",  "./imagesbooks/ukbench07588.jpg",  "./imagesbooks/ukbench07589.jpg",  "./imagesbooks/ukbench07590.jpg",  "./imagesbooks/ukbench08540.jpg",  "./imagesbooks/ukbench08542.jpg",  "./imagesbooks/ukbench08592.jpg"]




counter = 1 
for q_path in q_paths: 
    imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, 100)
    # imagematchesrgb , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, 0.7)

    # to reload module: uncomment use the following 
    # %load_ext autoreload
    # %autoreload 2

    import Accuracy as accuracy
    a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematchesrgb, 20)
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', i_rgb)


    # import ImageSearch_Plots as myplots
    # myplots.plot_predictions(imagematchesrgb, q_path)

    score = []
    matchesposition = [0]*len(imagematchesrgb)
    for i in i_rgb: 
        matchesposition[i] = 1

    # score, file = item
    for item in imagematchesrgb:
        x, y = item
        score.append(x)
    # print(score)

    import matplotlib.pyplot as plt 
    plt.scatter ( [counter]*len(imagematchesrgb), score, c=matchesposition)

    counter +=1 

plt.colorbar()
plt.show()



#### check Gaussian nature of scores and success positions
import ImageSearch_Algo_HSV
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import Accuracy as accuracy
from kneed import KneeLocator
import random 
import matplotlib.pyplot as plt

q_path = random.sample(q_paths, 1)[0]


# Features 
TESTNAME = 'Data519'
file_HSV_Feature = 'data/' + TESTNAME + '_PandasDF_HSV_Features'
file_HSV_Tree = 'data/' + TESTNAME + '_HSV_Tree'
file_RGB_Feature = 'data/' + TESTNAME + '_PandasDF_RGB_Features'
file_RGB_Tree = 'data/' + TESTNAME + '_RGB_Tree'
file_SIFT_Feature = 'data/' + TESTNAME + '_PandasDF_SIFT_Features_kp100'
mydataRGB = ImageSearch_Algo_RGB.RGB_LOAD_FEATURES (file_RGB_Feature)
mydataHSV = ImageSearch_Algo_HSV.HSV_LOAD_FEATURES (file_HSV_Feature)
mydataSIFT = ImageSearch_Algo_SIFT.SIFT_LOAD_FEATURES (file_SIFT_Feature)
# Tree & Clusters 
myRGBtree = ImageSearch_Algo_RGB.RGB_Load_Tree (file_RGB_Tree)
myHSVtree = ImageSearch_Algo_HSV.HSV_Load_Tree (file_HSV_Tree)


imagematches , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH_TREE (myRGBtree, mydataRGB, q_path, 100)

# imagematches , searchtimergb = ImageSearch_Algo_RGB.RGB_SEARCH(mydataRGB, q_path, 0.5)

# imagematches , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, 100)
# imagematches , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH(mydataHSV, q_path)


# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2


a, d, i_rgb, cnt = accuracy.accuracy_matches(q_path, imagematches, 20)
print ('Accuracy =',  a, '%', '| Quality:', d )
print ('Count', cnt, ' | position', i_rgb)


# import ImageSearch_Plots as myplots
# myplots.plot_predictions(imagematches, q_path)

score = []
successScore = []
# score, file = item
for item in imagematches:
    x, y = item
    score.append(x)
# print(score)
successPositions =i_rgb
for i in i_rgb: 
    successScore.append(score[i])

#  can throw exceptions in case of less points
try : 
    elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
    print ('Detected Elbow cluster value :', elbow.knee)
except: 
    pass    
qualifiedItems = min (elbow.knee, 6)

# plt.scatter ( [counter]*len(imagematches), score, c=matchesposition)
plt.plot(score)
plt.scatter(successPositions, successScore, c='r' )
plt.vlines( qualifiedItems , 0, 7, colors='g')




##############################################################################
#######################------  HSV THRESHOLDING------ ########################
##############################################################################




import ImageSearch_Algo_HSV

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = "./imagesbooks/"
imagepaths = list(paths.list_images(IMGDIR))
# print (imagepathss)

mydataHSV, mytime = ImageSearch_Algo_HSV.HSV_GEN(imagepaths)
print ('HSV Feature Generation time', mytime)

# to create a new tree from dataframe features 'mydataHSV'
mytree = ImageSearch_Algo_HSV.HSV_Create_Tree (mydataHSV, savefile='HSV_Tree')


#------SEARCH TEST------------------------------#

# q_path = random.sample(imagepaths, 1)[0]
# q_paths = random.sample(imagepaths, 20)  # random sample 100 items in list

q_paths = ["./imagesbooks/ukbench00196.jpg",  "./imagesbooks/ukbench00199.jpg",  "./imagesbooks/ukbench00296.jpg",  "./imagesbooks/ukbench00298.jpg",  "./imagesbooks/ukbench00299.jpg",  "./imagesbooks/ukbench00300.jpg",  "./imagesbooks/ukbench00302.jpg",  "./imagesbooks/ukbench00303.jpg",  "./imagesbooks/ukbench02730.jpg",  "./imagesbooks/ukbench02740.jpg",  "./imagesbooks/ukbench02743.jpg",  "./imagesbooks/ukbench05608.jpg",  "./imagesbooks/ukbench05932.jpg",  "./imagesbooks/ukbench05933.jpg",  "./imagesbooks/ukbench05934.jpg",  "./imagesbooks/ukbench05935.jpg",  "./imagesbooks/ukbench05952.jpg",  "./imagesbooks/ukbench05953.jpg",  "./imagesbooks/ukbench05954.jpg",  "./imagesbooks/ukbench05955.jpg",  "./imagesbooks/ukbench05956.jpg",  "./imagesbooks/ukbench05957.jpg",  "./imagesbooks/ukbench05958.jpg",  "./imagesbooks/ukbench05959.jpg",  "./imagesbooks/ukbench06148.jpg",  "./imagesbooks/ukbench06149.jpg",  "./imagesbooks/ukbench06150.jpg",  "./imagesbooks/ukbench06151.jpg",  "./imagesbooks/ukbench06558.jpg",  "./imagesbooks/ukbench06559.jpg",  "./imagesbooks/ukbench07285.jpg",  "./imagesbooks/ukbench07588.jpg",  "./imagesbooks/ukbench07589.jpg",  "./imagesbooks/ukbench07590.jpg",  "./imagesbooks/ukbench08540.jpg",  "./imagesbooks/ukbench08542.jpg",  "./imagesbooks/ukbench08592.jpg", "./imagesbooks/ukbench08594.jpg"
,"./imagesbooks/ukbench08595.jpg","./imagesbooks/ukbench08609.jpg","./imagesbooks/ukbench09364.jpg","./imagesbooks/ukbench09365.jpg","./imagesbooks/ukbench09366.jpg","./imagesbooks/ukbench10061.jpg","./imagesbooks/ukbench10065.jpg","./imagesbooks/ukbench10066.jpg","./imagesbooks/ukbench10085.jpg","./imagesbooks/ukbench10087.jpg","./imagesbooks/ukbench10108.jpg","./imagesbooks/ukbench10109.jpg","./imagesbooks/ukbench10110.jpg","./imagesbooks/ukbench10112.jpg","./imagesbooks/ukbench10113.jpg","./imagesbooks/ukbench10114.jpg","./imagesbooks/ukbench10116.jpg","./imagesbooks/ukbench10118.jpg","./imagesbooks/ukbench10119.jpg","./imagesbooks/ukbench10124.jpg","./imagesbooks/ukbench10125.jpg","./imagesbooks/ukbench10126.jpg","./imagesbooks/ukbench10128.jpg","./imagesbooks/ukbench10129.jpg","./imagesbooks/ukbench10130.jpg","./imagesbooks/ukbench10131.jpg","./imagesbooks/ukbench10152.jpg","./imagesbooks/ukbench10153.jpg","./imagesbooks/ukbench10154.jpg","./imagesbooks/ukbench10164.jpg","./imagesbooks/ukbench10165.jpg","./imagesbooks/ukbench10166.jpg","./imagesbooks/ukbench10167.jpg"]




counter = 1 
for q_path in q_paths: 
    imagematcheshsv , searchtimehsv = ImageSearch_Algo_HSV.HSV_SEARCH_TREE (myHSVtree, mydataHSV, q_path, 100)
       

    # to reload module: uncomment use the following 
    # %load_ext autoreload
    # %autoreload 2

    import Accuracy as accuracy
    a, d, i_hsv, cnt = accuracy.accuracy_matches(q_path, imagematcheshsv, 20)
    # print ('Accuracy =',  a, '%', '| Quality:', d )
    # print ('Count', cnt, ' | position', i_rgb)


    # import ImageSearch_Plots as myplots
    # myplots.plot_predictions(imagematchesrgb, q_path)

    score = []
    matchesposition = [0]*len(imagematcheshsv)
    for i in i_hsv: 
        matchesposition[i] = 1

    # score, file = item
    for item in imagematcheshsv:
        x, y = item
        score.append(x)
    # print(score)

    import matplotlib.pyplot as plt 
    plt.scatter ( [counter]*len(imagematcheshsv), score, c=matchesposition)

    counter +=1 

plt.colorbar()
plt.show()

# 


