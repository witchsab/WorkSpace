import PIL
from PIL import Image
import imagehash
import random
from pprint import pprint
from imutils import paths
import os
import time
import matplotlib.pyplot as plt
import dataAnalysisSearchService as search


# ALGO Module 
# This module contains all the algos to be used for the image search
# Input: Caller provides image 'ID' string to search for similar images in database 
# Output: return a sorted list of tuples (ID, Score) of the following format 
# sort: (by score: best -> worst)
# format: [( ID1, Score1),( ID2, Score2),( ID3, Score3), ... ]


# Function template 
# You can create separate files/modules per algo as needed
def algo1(image_id): 

    # Example output: Sample list of tuples 
    # image_id_list = [('125401',29), ('107001',30), (109802,77), (2,100)]

    image_id_list = search.searchNeedle(image_id)

    return image_id_list
