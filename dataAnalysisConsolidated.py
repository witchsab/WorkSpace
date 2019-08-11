
import os
import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
from imutils import paths

# for hash all the images in folder / database
import Accuracy as accuracy
import ImageSearch_Algo_Hash
import ImageSearch_Algo_HSV
import ImageSearch_Algo_ORB
import ImageSearch_Algo_RGB
import ImageSearch_Algo_SIFT
import ImageSearch_Plots as myplots

# --------------- VAR COMMONS------------------

IMGDIR = r'./imagesbooks/'
IMGDIR = r"V:\\Download\\imagesbooks\\"
IMGDIRPROCESSED = ['']*5
IMGDIRPROCESSED[0] = r"V:\\Download\\imagesbooks1\\"
IMGDIRPROCESSED[1] = r"V:\\Download\\imagesbooks2\\"
IMGDIRPROCESSED[2] = r"V:\\Download\\imagesbooks3\\"
IMGDIRPROCESSED[3] = r"V:\\Download\\imagesbooks4\\"
IMGDIRPROCESSED[4] = r"V:\\Download\\imagesbooks_warp\\"

# --------------- TEST PARAMETERS ----------------------#
TESTNAME = "Data519"

# ----------- GENERATE ALL FEATURES & SAVE --------------#

