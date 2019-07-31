
import os 
# import ImageSearch_Algo_Hash as ImageSearch_Algo_Hash
# import ImageSearch_Algo_RGB as ImageSearch_Algo_RGB
import pandas
from imutils import paths
import random 


# ------------- GENERATION TEST-------------------#

# Hyper-Parameter for comparing histograms
parametercorrelationthreshold = 0.70

IMGDIR = r"V:\\Download\\imagesbooks\\"
IMGDIRPROCESSED1 = r"V:\\Download\\imagesbooks1\\"
IMGDIRPROCESSED2 = r"V:\\Download\\imagesbooks2\\"
IMGDIRPROCESSED3 = r"V:\\Download\\imagesbooks3\\"
IMGDIRPROCESSED4 = r"V:\\Download\\imagesbooks4\\"


imagepaths = list(paths.list_images(IMGDIRPROCESSED2))
# print (imagepathss)

mydata, mytime = RGB_GEN(imagepaths)




#------SEARCH TEST------------------------------#

q_path = random.sample(imagepaths, 1)[0]
imagematches , searchtime = RGB_SEARCH(mydata, q_path, 0.8)

# to reload module: uncomment use the following 
# %load_ext autoreload
# %autoreload 2

import Accuracy as accuracy
a = accuracy.accuracy_matches(q_path, imagematches, 50)
print ('Accuracy =',  a, '%')


import ImageSearch_Plots as myplots
myplots.plot_predictions(imagematches[:20], q_path)


#---------------- Compile data and plot results 

accStats = pd.DataFrame(columns=['file','Acc', 'PCount'])

q_paths = random.sample(imagepaths, 100)  # random sample 100 items in list 

for q_path in q_paths:    
    imagematches , searchtime = RGB_SEARCH(mydata, q_path, 0.7)
    a = accuracy.accuracy_matches(q_path, imagematches, 50)
    accStats = accStats.append({ 'file': q_path, 'Acc': a, 'PCount': len(imagematches) } , ignore_index=True)


plt.plot(accStats['Acc'])
plt.plot (accStats['PCount'])
plt.hlines(accStats['Acc'].mean(), 0, 100, 'r')

print ("Mean Acc = ", accStats['Acc'].mean())