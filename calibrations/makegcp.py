# -*- coding: utf-8 -*-
"""
makegcp.py

This script generates Ground Control Points (GCPs) by clicking on the location 
of the gps device from image. It is suggested to run this as a BATCH script.

This script is run once only and returns a csv of GCPs per georeferencing campaign.
The outcome is highly dependent on the accuracy of the georeferencing campaign
carried out in-situ given the following strict requirements:
- with a GPS locator, namely on board of a boat in this case;
- making sure a snapshot of that measurement is recorded from camera.

@author: santinel
"""

import os, glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

PATH_IMAGES = [r'geooutput\TLC00009_00890_02300',
               r'geooutput\TLC00010_00001_03608',
               r'geooutput\TLC00011_00001_02836']

def createFolders(pathname):
    # create folders
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    return

outputcsv = r'geooutput/point_bridge_gcps.csv'
outputimages = r'geooutput/seeGCPs'

createFolders(outputimages)

images = list()
for p in PATH_IMAGES:
    images.extend(glob.glob(os.path.join(p,'*.jpg')))

Xgcp = list(); Ygcp = list(); fname = list()
# run ginput, save images.
for imm in images:
    imfloat = plt.imread(imm)
    fig = plt.figure(figsize=(10,8))
    ax = plt.imshow(imfloat)
    pp = plt.ginput(1, timeout=-1)
    plt.plot(pp[0][0], pp[0][1],'x',color='magenta', markersize=6, linewidth=3)
    # save them so to look at them later on.
    plt.savefig(os.path.join(outputimages,os.path.basename(imm)))
    plt.close()
    fname.append(os.path.basename(imm))
    Xgcp.append(pp[0][0])
    Ygcp.append(pp[0][1])
    print(pp)

# record data on csvfile # write csv
df = pd.DataFrame({'img_filename': fname, 'xgcp': Xgcp, 'ygcp': Ygcp})
df.to_csv(outputcsv, index=False)

# !TODO After running, get manually rid of points which have no reference.
