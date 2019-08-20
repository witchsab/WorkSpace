import pandas as pd
from kneed import KneeLocator

import Accuracy as accuracy

#######################################################################
# -----------  HELPER ALGOS    ------------ #


''' Knee thresholding on based on scores in imagematches '''

def autothreshold_Knee (imagematches , verbose=False):
    score = []
    # successScore = []
    filelist = []
    # score, file = item
    for item in imagematches:
        x, y = item
        score.append(x)
        filelist.append (y)
    # print(score)
    # successPositions =i_rgb
    # for i in i_rgb: 
    #     successScore.append(score[i])
    
    # setting default value  
    myknee = 0
    try: 
        elbow = KneeLocator( list(range(0,len(score))), score, S=2.0, curve='convex', direction='increasing')
        myknee = elbow.knee
        if verbose: 
            print ('Detected Elbow cluster value :', myknee)
    except: 
        print ('Error in AutoThresholding.')
        pass
    qualifiedItems = min (myknee , 6)
    return filelist[:qualifiedItems]





#######################################################################
# -----------  HELPER FUNCTIONS    ------------ #

'''
merges a list of imagematches to generate a mergelist and filters the input dataframe with matching mergelist
'''
def filter_SIFT_Candidates(listimagematches, mydataframe):
    # create a mergelist 
    mergelist = set ()

    # combine imagematch1, imagematch2, imagematch3 ...
    for imagematches in listimagematches: 
        # searchFile, searchResults = matcheshsv[0]
        for myitem in imagematches:
            x, y = myitem
            mergelist.add(y)
                
    mergelist = list(mergelist)
    # print ("Candidate count ", len(mergelist))
    # row_dict['mergecount'] = len(mergelist)
    filteredframe = mydataframe[mydataframe['file'].isin(mergelist)]
    return filteredframe


'''
Removes duplicates from list while preserving order 
Fast implementation (check stackoverflow)
'''
def sanitize_List(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


'''
Merges list if final candidates in the order of list 
1. Create a top list of common elements
2. Append remainder unique elements in list order  
seq is a list of lists
'''

import time

def merge_results(seq, verbose=False):

    start = time.time()
    mergedlist = []
    # contains the intersection elements
    result = set(seq[0])
    for s in seq[1:]:
        result.intersection_update(s)

    # added the common element to beginning of list 
    mergedlist = list (result)

    if verbose: print ('Len Common', len(mergedlist)) , print (" Commons ", mergedlist)
    # appending remaining element to mergelist preserving order 
    for item in seq :
        mergedlist = mergedlist + item
    
    #  print in verbose mode 
    if verbose: 
        print ("Total", mergedlist)    
        t = time.time() - start 
        print ('Merge Time ', t ) 
    
    return mergedlist

    # removing duplicates
    seen = set()
    seen_add = seen.add
    finalList = [x for x in mergedlist if not (x in seen or seen_add(x))]


    return finalList

'''
convert imagepredictions to a list of image file names/ids 
'''
def getListfromImagepredictions (imagepredictions):
    imagelist = []
   # append top 20 results from SIFT to end of TOPLIST 
    for myitem in imagepredictions:
        x, y = myitem
        imagelist.append(y)
    return imagelist
