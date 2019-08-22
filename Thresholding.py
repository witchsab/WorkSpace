"""
This Module contains all the thresholding algo varianrs and some helper functions. 
"""

import time

import pandas as pd
from kneed import KneeLocator

import Accuracy as accuracy


#######################################################################
# -----------  HELPER ALGOS    ------------ #



def autothreshold_knee(imagematches, cutoff=6, verbose=False):
    """ Knee thresholding on scores in imagematches """

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
        elbow = KneeLocator(list(range(0, len(score))), score, S=2.0, curve='convex', direction='increasing')
        myknee = elbow.knee
        if verbose:
            print('Detected Elbow cluster value :', myknee)
    except:
        print('Error in AutoThresholding.')
        pass

    qualified_items = min(myknee, cutoff)

    return filelist[:qualified_items]





#######################################################################
# -----------  HELPER FUNCTIONS    ------------ #


def filter_sift_candidates(listimagematches, mydataframe):
    """Merges a list of imagematches to generate a mergelist and
    filters the input dataframe with matching mergelist"""

    # create a mergelist
    merge_list = set ()

    # combine imagematch1, imagematch2, imagematch3 ...
    for imagematches in listimagematches:
        # searchFile, searchResults = matcheshsv[0]
        for myitem in imagematches:
            x, y = myitem
            merge_list.add(y)

    merge_list = list(merge_list)
    # print ("Candidate count ", len(mergelist))
    # row_dict['mergecount'] = len(mergelist)
    filteredframe = mydataframe[mydataframe['file'].isin(merge_list)]
    return filteredframe


def filter_candidates(listimagematches, mydataframe):
    """Merges a list of imagematches to generate a mergelist and
    filters the input dataframe with matching mergelist"""

    # create a mergelist
    merge_list = set ()

    # combine imagematch1, imagematch2, imagematch3 ...
    for imagematches in listimagematches:
        # searchFile, searchResults = matcheshsv[0]
        for myitem in imagematches:
            x, y = myitem
            merge_list.add(y)

    merge_list = list(merge_list)
    # print ("Candidate count ", len(mergelist))
    # row_dict['mergecount'] = len(mergelist)
    filteredframe = mydataframe[mydataframe['file'].isin(merge_list)]
    return filteredframe




def sanitize_list(seq):
    """
    Removes duplicates from list while preserving order
    Fast implementation (check stackoverflow)
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def merge_results(seq, verbose=False):
    """
    Merges list if final candidates in the order of list
    1. Create a top list of common elements
    2. Append remainder unique elements in list order
    Remember: seq is a list of lists
    """
    start = time.time()
    merged_list = []
    # contains the intersection elements = common to all
    result = set(seq[0])
    for s in seq[1:]:
        result.intersection_update(s)

    # added the common element to beginning of list
    merged_list = list (result)

    if verbose:
        print('Len Common', len(merged_list)) 
        print('Commons ', merged_list)

    # appending remaining element to mergelist preserving order
    for item in seq:
        merged_list = merged_list + item

    #  print in verbose mode
    if verbose:
        print("Total", merged_list)
        t = time.time() - start
        print('Merge Time ', t)

    # return mergedlist

    # removing duplicates
    seen = set()
    seen_add = seen.add
    final_list = [x for x in merged_list if not (x in seen or seen_add(x))]

    return final_list


def imagepredictions_to_list (imagepredictions):
    """
    convert imagepredictions to a list of image file names/ids
    """
    imagelist = []
   # append top 20 results from SIFT to end of TOPLIST
    for myitem in imagepredictions:
        x, y = myitem
        imagelist.append(y)
    return imagelist
