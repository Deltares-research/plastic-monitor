# -*- coding: utf-8 -*-
"""
Authors: Jonathan Nuttall, Giorgio Santinelli
Organisation: Deltares
Contact: marieke.eleveld@deltares.nl
Remarks
"""

import numpy as np
import cv2
import pims
import ctypes
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from tqdm import tqdm
import colorsys
import time
import sys

# add calibration folder to import path
sys.path.append('../calibrations')

import georef as gr
import lenses as ls

lenses = ls.lenses()
georef = gr.georef()

# Initial Font text
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100, 100)
fontScale = 1
fontColor = (0, 255, 255)
thickness = 2
lineThickness = 2

# Initialize windows for dialogue box (Open AVI file)
root = tk.Tk()
root.withdraw()

ix = -1
iy = -1
cx = -1
cy = -1
noClicks = 0

class debris:
    '''
        debris class: debris statistics for video
    '''


    def __init__(self, frame_rate):
        self.frame_rate = frame_rate
        self.debris = pd.DataFrame({"Frame": [], "Centroid": [], "Area": [], "Organic": []})

    def add(self, frame, stats, centroids, pixel_area, organics):
        '''
            draw and record a vertical line on a frame using a single coordinate,
            adjusted for the calibration of the camera, i.e. fisheye lenses

            Parameters
            ----------
            frame : array : frame from video
            stats : array : statistics for objects (opencv2: see cv2.connectedComponentsWithStats)
            centroids : array: centroid of all objects corresponding to stats
            pixel_area: array: average pixel to real world area for object in stats (adjusted for lens)
            organics: bool: if an object was judge to be organic

            Returns
            -------
            None
        '''
        new_debris = {"Frame": [], "Centroid": [], "Area": [], "Organic": []}
        pixel_area = np.array(pixel_area)
        for area, pixel, centre, organic in zip(stats[1:, 4], pixel_area, centroids[1:], organics):
            new_debris['Frame'].append(frame)
            new_debris['Centroid'].append(centre)
            new_debris['Area'].append(area * pixel)
            new_debris['Organic'].append(organic)

        self.debris = self.debris.append(pd.DataFrame(new_debris), ignore_index=True)


    def freqTime(self):
        '''
        Calculates frequency of debris objects from stats

        Returns
        -------
        float: average number of objects per second
        '''
        amount = self.debris.groupby('Frame').size()
        return amount/2.0


    def freqOrganicsTime(self):
        '''
            Calculates frequency of organic debris objects from stats

            Returns
            -------
            float: average number of objects per second
        '''
        filter_cond = self.debris['Organic'] == True
        amount = self.debris.where(filter_cond).groupby('Frame').size()
        return amount/2.0


    def areaTime(self):
        '''
            Calculates area of debris objects from stats

            Returns
            -------
            float: average number of objects per second
        '''
        return self.debris.groupby('Frame')['Area'].apply(lambda x: x.sum()/2.0)

    def areaOrganicsTime(self):
        '''
            Calculates area of organic debris objects from stats

            Returns
            -------
            float: average number of objects per second
        '''
        filter_cond = self.debris['Organic'] == True
        return self.debris.where(filter_cond).groupby('Frame')['Area'].apply(lambda x: x.sum() / 2.0)


def corrected_len_horizontal_line(frame, pointx, line_color, thickness, args):
    '''
        draw and record a horizontal line on a frame using a single coordinate,
        adjusted for the calibration of the camera, i.e. fisheye lenses

        Parameters
        ----------
        frame : array : frame from video
        pointx : array : (x, y) coordinates
        line_color : array: (B(lue), G(reen), R(ed)) colour of line
        thickness: int: line thickness
        args: H, K and Distortion (from camera calibration)

        Returns
        -------
        frame: array: frame with drawn line
        points: array: (x,y) coords alone the line [100 points]

    '''


    H, K, distortion = args
    Ucoord = np.array([0, 2000])
    Vcoord = np.array([pointx[1], pointx[1]])
    Xcoord, Ycoord = georef.UV2XYZ(Ucoord, Vcoord, H, K, distortion)
    Xcoord_array = np.linspace(Xcoord[0], Xcoord[1], 50)
    Ycoord_array = np.linspace(Ycoord[0], Ycoord[1], 50)
    Ucoord_array, Vcoord_array = georef.XYZ2UV(Xcoord_array, Ycoord_array, H, K, distortion)
    coords = lambda xx, yy: [(int(x), int(y)) for x, y in zip(xx, yy)]
    points = coords(list(Ucoord_array), list(Vcoord_array))
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, line_color, thickness)
    return frame, points


def corrected_len_vertical_line(frame, pointx, line_color, thickness, args):
    '''
        draw and record a vertical line on a frame using a single coordinate,
        adjusted for the calibration of the camera, i.e. fisheye lenses

        Parameters
        ----------
        frame : array : frame from video
        pointx : array : (x, y) coordinates
        line_color : array: (B(lue), G(reen), R(ed)) colour of line
        thickness: int: line thickness
        args: H, K and Distortion (from camera calibration)

        Returns
        -------
        frame: array: frame with drawn line
        points: array: (x,y) coords alone the line [100 points]

    '''

    H, K, distortion = args
    Ucoord = np.array([pointx[0], pointx[0]])
    Vcoord = np.array([0, 1000])
    Xcoord, Ycoord = georef.UV2XYZ(Ucoord, Vcoord, H, K, distortion)
    Xcoord_array = np.linspace(Xcoord[0], Xcoord[1], 100)
    Ycoord_array = np.linspace(Ycoord[0], Ycoord[1], 100)
    Ucoord_array, Vcoord_array = georef.XYZ2UV(Xcoord_array, Ycoord_array, H, K, distortion)
    coords = lambda xx, yy: [(int(x), int(y)) for x, y in zip(xx, yy)]
    points = coords(list(Ucoord_array), list(Vcoord_array))
    cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, line_color, thickness)
    return frame, points


def is_validframe(cap, frame_index):

    '''
        get valid frames from a video,

        Parameters
        ----------
        cap : opencv2 video capture object
        frame_index : frame index

        Returns
        -------
        idxs: bool: not None frame if True

    '''
    flag = False
    # set the frame id to read that particular frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if frame is not None:
        flag = True
    return flag


def get_background(cap):

    '''
        calculate median frame from 50 random frames of video

        Parameters
        ----------
        cap : opencv2 video capture object

        Returns
        -------
        median_frame: array: median background image

    '''

    # we will randomly select 50 frames for the calculating the median
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
    # we will store the frames in array
    frames = []
    for idx in frame_indices:
        # set the frame id to read that particular frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

def mouseCallBack(event, x, y, flags, param):
    '''
        register mouse clicks events and coordinates from opencv2 window.
        sets global coordinates

        Parameters
        ----------
        event: cv2 object - type of mouse click even
        x, y: int  (pixel based x, y, coordinates clicked(
        flag: unused
        params: unused

        Returns
        -------
        None

    '''

    global ix, iy, cx, cy, noClicks
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        noClicks += 1
    if event == cv2.EVENT_MOUSEMOVE:
        cx = x
        cy = y

def Mbox(title, text, style):

    '''
        opens an observing window to draw lines to highlight zones of interest

        Parameters
        ----------
        title : message box title
        text : text to present
        style: int style of box

        Returns
        -------
        ctypes Message box (to screen)

    '''
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def openWindowClicks(name, frame, args, horizontal=True):
    '''
        opens an observing window to draw lines to highlight zones of interest

        Parameters
        ----------
        name : string: name of window
        frame : array: single frame from openCV open video file
        args: H, K and Distortion (from camera calibration)
        horizontal: bool (optional): if the lines are to be horizontal

        Returns
        -------
        coords: array : array of coordinates along the drawn lines.

    '''

    global noClicks, iy, ix, cx, cy
    noClicks = 0

    if args is None:
        raise Exception("Distortion \ Lense Argurments not found")

    # Green color in BGR
    color = (0, 255, 0)
    # Line thickness of 9 px
    thickness = 2

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, mouseCallBack)

    while noClicks < 1:
        frame_copy = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

        if horizontal:
            frame_copy, coords = corrected_len_horizontal_line(frame_copy, [cx, cy], color, thickness, args)
            cv2.imshow(name, frame_copy)
        else:
            frame_copy, coords = corrected_len_vertical_line(frame_copy, [cx, cy], color, thickness, args)
            cv2.imshow(name, frame_copy)

    return coords


def mask_detection_zone(cap, frame_rate, args):
    '''

        generates mask to remove areas outside of a 2 second control zone across the river, based on frame_rate
        and flow rate. Flow rate being determined by the tracking manually of a single object in the water by clicking
        on images x frames apart based on the frame rate.

        Parameters
        ----------
        cap : captured video from opencv2
        frame_rate: frames per seconds
        args: H, K and Distortion (from camera calibration)

        Returns
        -------
        mask: image mask to produce measurement zone of interest across the river.

    '''

    global ix, iy



    idx = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(low=0.25, high=0.75)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#    if is_validframe(cap,idx):
#        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    coords_start = openWindowClicks("Select object in Frame n:", frame, args=args)

    frames_gap = max(round(2*frame_rate), 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx+frames_gap)
#    if is_validframe(cap,idx+frames_gap):
#        cap.set(cv2.CAP_PROP_POS_FRAMES, idx+frames_gap)
    ret, frame = cap.read()
    coords_end = openWindowClicks("Select same object in Frame n + (2 seconds):", frame, args=args)

    cv2.destroyAllWindows()

    coords_end.reverse()
    coords_start.extend(coords_end)
    height, width, channels = frame.shape
    mask = np.zeros((height, width), np.uint8)

    points = [i for i in coords_start if 0 <= i[0] <= width-1 and
              0 <= i[1] <= height-1]

    boundary = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [boundary], 255)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Selected Zone of Interest", res)
    result = Mbox('Continue ...... ', 'Is this selection Correct ?', 4)
    if result == 7:  # no
        ix = -1
        iy = -1
        mask, boundary = mask_detection_zone(cap, frame_rate, args)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

    return mask, boundary


def mask_river_banks(cap, args):
    '''
        generates mask to remove the banks from the video frames

        Parameters
        ----------
        cap : captured video from opencv2
        args: H, K and Distortion (from camera calibration)

        Returns
        -------
        mask: image mask of removing banks from the river.

    '''

    global ix, iy
    idx = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(low=0.25, high=0.75)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#    if is_validframe(cap,idx):
#        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    # left bank
    coords_left = openWindowClicks("Select Left Bank", frame, args=args, horizontal=False)
    cv2.destroyAllWindows()

    # right bank
    coords_right = openWindowClicks("Select Right Bank", frame, args=args, horizontal=False)
    cv2.destroyAllWindows()

    coords_right.reverse()
    coords_left.extend(coords_right)
    height, width, channels = frame.shape
    mask = np.zeros((height, width), np.uint8)
    boundary = np.array([coords_left], dtype=np.int32)
    cv2.fillConvexPoly(mask, boundary, len(coords_left), 255)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Image without bank", res)
    result = Mbox('Continue ...... ', 'Is this selection Correct ?', 4)
    if result == 7:  # no
        ix = -1
        iy = -1
        mask = mask_river_banks(cap, args)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
    return mask

def isOrganic(pixel_color):
    '''
        Computes if the colour of the pixel is indicative of an organic
        (e.g. a green\yellow\brown colour)

        Parameters
        ----------
        pixel colour : (1 x 3 array) -  b(lue) g(reen) r(ed) (opencv2 standard)

        Returns
        -------
        bool : True if the pixel is in the correct HSL range brown - green

    '''

    b, g, r = pixel_color
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    if 0.1 <= l <= 0.5 and 0.35 <= s <= 0.5 and 0.1 <= h <= 1/3:
        return True
    else:
        return False

if __name__ == "__main__":
    '''
    
        main code - "debrisDetection"

        Undertakes object detection on videos using color analysis of adjacent frames

    '''

    # Used to correct for setup\take down time of camera
    startFrame = 0
    endFrame = -1

    frame_rate = 1  # a frames per second
    filename = filedialog.askopenfilename(filetypes=[('.avi', '.avi')], title='Select *.avi video for processing')
    outputVideoFile = filename.replace('.avi', '_debris.avi')

    # Get distortion data
    H_distortion_path = Path(filedialog.askopenfilename(filetypes=[('.npy', '.npy')],
                                                        title='Select H calibration matrix file *.npy'))
    Calibration_params = Path(filedialog.askdirectory(title='Select camera setup folder'))
    
    georef.PATH_GEOOUT = H_distortion_path.parent.parent.absolute()
    georef.project = H_distortion_path.parent.parts[-1]
    H = np.load(H_distortion_path)
    K, distortion = lenses.readParams(Calibration_params.parent.absolute(), Calibration_params.parts[-1])

    cap = cv2.VideoCapture(filename)
    # get the video frame height and width
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # get real world coordinates from frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
#    if is_validframe(cap,cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()

    X, Y = georef.transformUV2XYZimage(frame, H, K, distortion)  # Frame grid in meters (meshgrid)

    # set up 2 frame zone (i.e. 2 second time interval)
    args = H, K, distortion
    zone_mask, outline_points = mask_detection_zone(cap, frame_rate, args)

    minY = np.min(np.array(outline_points)[:, 1]) - 1
    maxY = np.max(np.array(outline_points)[:, 1]) + 1

    bank_mask = mask_river_banks(cap, args)
    full_mask = cv2.bitwise_and(zone_mask, bank_mask, 0)[minY:maxY, :]

    # get the background model
    background = get_background(cap)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)[minY:maxY, :]
    background = cv2.bitwise_and(background, background, mask=full_mask)

    # rawframes = pims.Video(filename)
    rawframes = pims.PyAVVideoReader(filename)
    frames = rawframes[startFrame:endFrame]

    diffFrames = []
    print("Reading and Processing Frames.....\n==============================================", flush=True)
    time.sleep(1)
    for ind1, frame2 in tqdm(enumerate(frames[1:]), total=len(frames[1:])):
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)[minY:maxY, :]
        gray2 = cv2.bitwise_and(frame2, frame2, mask=full_mask)
        diff2 = cv2.absdiff(gray2, background)

        frame1 = cv2.cvtColor(frames[ind1], cv2.COLOR_BGR2GRAY)[minY:maxY, :]
        gray1 = cv2.bitwise_and(frame1, frame1, mask=full_mask)
        diff1 = cv2.absdiff(gray1, background)

        movingDiff = cv2.absdiff(diff1, diff2)

        #  calibrate threshold to image conditions to detect all debris both light and dark,
        #  difficult to collaborate in these conditions (sun and water reflection).

        thresholdDiff = cv2.inRange(movingDiff, 40, 255, 0)
        thresholdDiff = cv2.bitwise_and(thresholdDiff, thresholdDiff, cv2.THRESH_BINARY)

        if ind1 != 0:
            subDiff = cv2.bitwise_and(thresholdDiff, oldDiff)
            diffFrames.append(subDiff)

        oldDiff = thresholdDiff.copy()

    height, width, layer = rawframes[-1].shape
    size = (width, height)

    out = cv2.VideoWriter(outputVideoFile, cv2.VideoWriter_fourcc(*'XVID'), 15, size)
    debris_stats = debris(frame_rate)

    print("Writing Video and Collating Statistics.....\n==============================================", flush=True)
    print("Writing Output Video File to:", outputVideoFile, flush=True)
    time.sleep(1)

    color_organic = [0, 255, 0]
    color_non = [0, 0, 255]

    rawframes = pims.PyAVVideoReader(filename)
    for i, (diffFrame, rawFrame) in tqdm(enumerate(zip(diffFrames, rawframes[startFrame+1:endFrame-1])),
                                         total=len(diffFrames)):

        # Threshold it so it becomes binary
        ret, thresh = cv2.threshold(diffFrame, 0, 255, cv2.THRESH_BINARY)
        # Perform the operation
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        area = []
        organic = []
        for label in range(1, n_labels):
            centroid = centroids[label]
            centreX = round(centroid[0])
            centreY = round(centroid[1])

            # get average pixel size:
            localX = X[centreY, centreX-1:centreX+2]
            localY = Y[centreY + minY-1:centreY + minY+2, centreX]
            avXSize = np.abs(localX[2] - localX[0])/2
            avYSize = np.abs(localY[2] - localY[0])/2
            area.append(avXSize * avYSize)

            label_locations = np.where(labels == label)
            pixel_color = [0, 0, 0]
            for yloc, xloc in zip(label_locations[0], label_locations[1]):
                pixel_color += rawFrame[yloc + minY, xloc]
            pixel_color = pixel_color / len(label_locations[0])
            organic.append(isOrganic(pixel_color))
            if organic[-1]:
                color = color_organic
            else:
                color = color_non

            rawFrame = cv2.circle(rawFrame, (centreX, centreY + minY), 10, color, 2)

        rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2RGB)
        rawFrame = cv2.polylines(rawFrame, [np.array(outline_points)], True, (0, 255, 0), 2)
        debris_stats.add(i, stats, centroids, area, organic)

        rawFrame = cv2.putText(rawFrame, "Frame:" + str(i),
                               bottomLeftCornerOfText,
                               font,
                               fontScale,
                               fontColor,
                               thickness,
                               lineThickness)
        out.write(rawFrame)

    out.release()

    freq = debris_stats.freqTime()
    freq_organics = debris_stats.freqOrganicsTime()
    area = debris_stats.areaTime()
    area_organics = debris_stats.areaOrganicsTime()

    #Plotting Statistics

    # Plot objects in frequency per second
    ax = freq.plot(title="Objects per second", label="All Objects")
    freq_organics.plot(ax=ax, label="Organic Objects")
    ax.set_xlabel("Time(s)/Frame")
    ax.set_ylabel("Objects per second")
    ax.legend()
    plt.show()

    #plot object area in pixels per second
    area.plot(title="Area per second", label="All Objects")
    area_organics.plot(ax=ax, label="Organic Objects")
    ax.set_xlabel("Time(s)/Frame")
    ax.set_ylabel("Area of Objects per second ($m^2$)")
    ax.legend()
    plt.show()

    print("Statistics")
    print("===========================")
    print("Total Number of Frames:", len(diffFrames))
    print("Frame rate used in calculations:", frame_rate)
    print(" ")
    print("Total Number of Debris Items Detected:", sum(freq))
    print("Total Number of Organic Debris Items Detected:", sum(freq_organics))
    print(" ")
    print("Total Area of Debris Detected:", sum(area), "m²")
    print("Total Area of Organic Debris Detected:", sum(area_organics))
    print(" ")
    print("Max Number of Debris per second:", max(freq))
    print("Max Number of Organic Debris per second:", np.max(freq_organics))
    print(" ")
    print("Max Area of Organic Debris per second:", np.max(area_organics), "m²")
    print("Max Area of Debris per second:", max(area), "m²")
