

import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import random
import os 

'''
This function plots a list of tuples containing scores and paths
matches:    list of tuples (score int, filepath)
'''
def plot_predictions (matches,query_image_path ):
    #------------Generating result as p, d (redundant)------------#
    d = [] 
    p = []
    
    for item in matches: 
        score , thispath = item 
        d.append(str(thispath))
        p.append(score)

    #-----Plotting query image
    simg = plt.imread(query_image_path)
    plt.imshow(simg)
    plt.show()
    # d => list of image paths 
    # d = 

    # p => list of corresp scores 

    # Plotting all matches

    columns = 5 if len(d) >=5 else len (d)
    rows = int (len(d) / 5) + 1

    l = 0
    fig=plt.figure(figsize=(rows*4, columns*4))
    # ax enables access to manipulate each of subplots
    ax = []

    # print (len (d), rows, columns)  # test code 

    for i in range(1,  min( len(d) +1   ,   columns*rows +1)) :
        img = plt.imread(d[l])
        name = os.path.basename(d[l])
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title(name + '|  score='+str(p[l]))
        plt.imshow(img)
        l +=1
    plt.show()

