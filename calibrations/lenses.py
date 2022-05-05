# -*- coding: utf-8 -*-
"""
lenses.py

This set of functions cater of calibration of lense distortion in cameras mounted on
several location close to Bandung (Indonesia).
OpenCV default model and fisheye model can be run, which will return and save the
calibration parameters (mtx, dist), also called (K, dist) elsewhere.

default model: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
fisheye model: https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html

Functions such as distortPoints and undistortPoints can be used as conversion
functions to go from original pixel reference to undistorted pixel reference (undistortPoints)
and viceversa. 

Created on Wed Oct 27 18:01:28 2021

@author: santinel
"""

#calibration caltech test

import numpy as np
import cv2 as cv
import os, glob


class lenses():
    
    def __init__(self, 
                 PATH_CAL=r"c:\Users\santinel\Stichting Deltares\Robyn Gwee - videos\site2_bridge\20211020\splitframes\Bridge_20211020_cal2", 
                 PATH_CALOUT = r"caloutput",
                 project='Bridge_20211020_cal2',
                 model='fisheye'
                 ):
        # initialize paths
        self.PATH_CAL = PATH_CAL
        self.PATH_CALOUT = PATH_CALOUT
        self.project = project
        

    def createFolders(self, pathname):
        # create folders
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        return
    
    
    # reading K and dist
    def readParams(self, PATH_PARAMS, project):
        '''
        get distortion parameters from folders
        '''
        # retrieve undistort parameters
        K = np.load(os.path.join(PATH_PARAMS, project, 'matrix.npy'))
        distortion = np.load(os.path.join(PATH_PARAMS, project, 'dist.npy'))
        return K, distortion
    
    
    def array2openCV(self, array):
        '''
        OpenCV functions accepts array of type np.array([[[1.,2.],[4.,5.],[3.,6.]]])
        However a simpler way to ingest data would be providing the input as:
        np.array([[1.,2.],[4.,5.],[3.,6.]])
    
        Parameters
        ----------
        array : numpy array
            numpy array accepted by OpenCV, shape = (3,2)
    
        Returns
        -------
        numpy array as e.g. np.array([[1.,2.],[4.,5.],[3.,6.]])
        '''
        if len(array.shape) == 2 and array.shape[1] == 2:
            array = array.reshape(-1,1,2)
        else: 
            print('input shape should be (nrows,2)')
        return array
        
    def openCV2array(self, array):
        '''
        OpenCV functions accepts array of type np.array([[[1.,2.],[4.,5.],[3.,6.]]])
        However a simpler way to ingest data would be providing the input as:
        np.array([[1.,2.],[4.,5.],[3.,6.]])
    
        Parameters
        ----------
        array : numpy array
            numpy array as e.g. np.array([[1.,2.],[4.,5.],[3.,6.]])
    
        Returns
        -------
        numpy array accepted by OpenCV
        '''
        if len(array.shape) == 3 and array.shape[2] == 2:
            array = array.reshape(-1,2)
        else: 
            print('input shape should be (nrows,1,2)')
        return array
    
    
    def distortPoints(self, pointscustom, mtx, dist):
        '''
        Distort camera coordinate points from an undistorted plane coordinate reference.
    
        Parameters
        ----------
        pointscustom : numpy array
            Array of pixels coming from the undistorted image.
            e.g. np.array([[[1.,2.],[4.,5.],[3.,6.]]])
        mtx : array
            calibration matrix K
        dist : array
            calibration distortion vector dist
    
        Returns
        -------
        distorted : numpy array
            array of pixels from original (=distorted) image
            e.g. np.array([[[1.,2.],[4.,5.],[3.,6.]]])
    
        '''
        kInv = np.linalg.inv(mtx)
        pointsundist = np.zeros(pointscustom.shape)
        for i in range(len(pointsundist)):
            srcv = np.array([pointscustom[i][0][0], pointscustom[i][0][1], 1])
            dstv = kInv.dot(srcv)
            pointsundist[i][0][0] = dstv[0]
            pointsundist[i][0][1] = dstv[1]
        distorted = cv.fisheye.distortPoints(pointsundist, mtx, dist)
        return distorted
    
    
    def undistortPoints(self, pointscustom, mtx, dist):
        '''
        Undistort camera coordinate points from a distorted (original) camera coordinate reference.
        
        Parameters
        ----------
        pointscustom : numpy array
            Array of pixels coming from the undistorted image.
            e.g. np.array([[[1.,2.],[4.,5.],[3.,6.]]])
        mtx : array
            calibration matrix K
        dist : array
            calibration distortion vector dist
    
        Returns
        -------
        undistorted : numpy array
            array of pixels from undistorted image
            e.g. np.array([[[1.,2.],[4.,5.],[3.,6.]]])
    
        '''
        undistorted_points = cv.fisheye.undistortPoints(pointscustom, mtx, dist)
        undistorted_points = undistorted_points.reshape(-1,2)
        
        #newcameramtx = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (1920, 1080), np.eye(3), balance=1.)
        newcameramtx = mtx
        fx = newcameramtx[0,0]
        fy = newcameramtx[1,1]
        cx = newcameramtx[0,2]
        cy = newcameramtx[1,2]
        undistorted_array = np.zeros_like(undistorted_points)
        for i, (x, y) in enumerate(undistorted_points):
            px = x*fx + cx
            py = y*fy + cy
            undistorted_array[i,0] = px
            undistorted_array[i,1] = py
        undistorted = undistorted_array.reshape(-1,1,2)
        return undistorted
    
    
    def default_model(self):
        '''
        default_model
        
        Run model for calibration of default lenses images, using checkerboard images,
        with 7*9 checkerboard size.
    
        Parameters
        ----------
        PATHCAL    : path where images are
    
        Returns
        -------
        detection of checkerboard crossings;
        (mtx, dist) calibration parameters;
        undistortion of images as a check.
        '''
        
        
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # using all checkerboard crossing will work better
        objp = np.zeros((7*9,3), np.float32) #(6*7)
        objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2) #[0:7,0:6]
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(os.path.join(self.PATH_CAL,'*.jpg'))
        
        for fname in images:
            print(f'reading {os.path.basename(fname)}')
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (9,7), None) #(7,6)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (6,6), (-1,-1), criteria) #(6,6)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (9,7), corners2, ret) #(7,6)
                #cv.imshow('img', img)
                cv.imwrite(os.path.join(self.PATH_CALOUT,'chessboardcorners',
                                        self.project,os.path.basename(fname)), img)
                cv.waitKey(500)
        cv.destroyAllWindows()
        
        print(f'{np.shape(objpoints)[0]} images are used for calibration.')
        # run calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # export calibration parameters as csv # mtx, dist
        path_mtx = os.path.join(self.PATH_CALOUT, 'parameters', self.project, 'matrix.npy')
        path_dist   = os.path.join(self.PATH_CALOUT, 'parameters', self.project, 'dist.npy')
        with open(path_mtx, 'wb') as f:
            np.save(f, mtx)
        with open(path_dist, 'wb') as f:
            np.save(f, dist)
        
        # undistort
        for fname in images:
            print(f'undistorting {os.path.basename(fname)}')
            img = cv.imread(fname)
            h,  w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            # undistort
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image. Without it, f.o.v. corners will look too stretched,
            # and the camera calibration model can't resolve those corners.
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv.imwrite(os.path.join(self.PATH_CALOUT,'undistort',
                                    self.project, os.path.basename(fname)), dst)
        
        # print errors
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("Total error: {}".format(mean_error/len(objpoints)))
        
        return
    
    
    def fisheye_model(self):
        '''
        fisheye_model
        
        Run model for calibration of fisheye lenses images, using checkerboard images,
        with 7*9 checkerboard size.
    
        Parameters
        ----------
        PATHCAL    : path where images are
    
        Returns
        -------
        detection of checkerboard crossings;
        (mtx, dist) calibration parameters;
        undistortion of images as a check.
        '''
        
        
        # Checkboard dimensions
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # using all checkerboard crossing will work better
        #objp = np.zeros((7*9,3), np.float32) #(6*7)
        #objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2) #[0:7,0:6]
        objp = np.zeros((1, 7*9, 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)
        
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        images = glob.glob(os.path.join(self.PATH_CAL,'*.jpg'))
        
        for fname in images:
            print(f'reading {os.path.basename(fname)}')
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (7,9), cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE) #(7,6)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (6,6), (-1,-1), criteria) #(6,6)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (9,7), corners2, ret) #(7,6)
                #cv.imshow('img', img)
                cv.imwrite(os.path.join(self.PATH_CALOUT,'chessboardcorners_fisheye',
                                        self.project,os.path.basename(fname)), img)
                cv.waitKey(500)
        cv.destroyAllWindows()
        
        print(f'{np.shape(objpoints)[0]} images are used for calibration.')
        # run calibration
        K = np.zeros((3, 3)); D = np.zeros((4, 1))
        N_OK = len(objpoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        retval, mtx, dist, rvecs, tvecs = cv.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1],
                                    K, D, rvecs, tvecs, flags=calibration_flags,
                                    criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        
        # export calibration parameters as csv # mtx, dist
        path_mtx = os.path.join(self.PATH_CALOUT, 'parameters_fisheye', self.project, 'matrix.npy')
        path_dist   = os.path.join(self.PATH_CALOUT, 'parameters_fisheye', self.project, 'dist.npy')
        with open(path_mtx, 'wb') as f:
            np.save(f, mtx)
        with open(path_dist, 'wb') as f:
            np.save(f, dist)
        
        # undistort as a check
        for fname in images:
            print(f'undistorting {os.path.basename(fname)}')
            img = cv.imread(fname)
            h,  w = img.shape[:2]
            scaled_K = mtx
            
            scale_factor = 1.0 #DIM = [1920, 1080]
            balance = 1.0 # balance is in [0,1] range.
            DIM =(int(w*scale_factor), int(h*scale_factor))
            if scale_factor != 1.0:
                scaled_K = mtx*scale_factor
                scaled_K[2][2] = 1.0
                
            newcameramtx = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
                scaled_K, dist, DIM, np.eye(3), balance=balance)
            # init undistort
            map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, dist, np.eye(3),
                newcameramtx, DIM, cv.CV_16SC2)
            # undistort
            undist_image = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT)
            # save image
            cv.imwrite(os.path.join(self.PATH_CALOUT,'undistort_fisheye',
                                    self.project,os.path.basename(fname)), undist_image)
        
        return
    
    def run_calibration(self):
        '''
        Run model for calibration, using checkerboard images
    
        Parameters
        ----------
        model : string
            MODEL can be either 'default' (no fish-eye) or 'fisheye'.
            fisheye is better capable in mapping close-to-edge pixels in 
            images taken with large distortion lenses. 
    
        Returns
        -------
        calibration parameters saved in folder
    
        '''
        
        if self.model=='fisheye':
            # general cal output folder
            self.createFolders(os.path.join(self.PATH_CALOUT,'chessboardcorners_fisheye'))
            self.createFolders(os.path.join(self.PATH_CALOUT,'undistort_fisheye'))
            self.createFolders(os.path.join(self.PATH_CALOUT,'parameters_fisheye'))
            self.createFolders(os.path.join(self.PATH_CALOUT,'chessboardcorners_fisheye',self.project))
            self.createFolders(os.path.join(self.PATH_CALOUT,'undistort_fisheye',self.project))
            self.createFolders(os.path.join(self.PATH_CALOUT,'parameters_fisheye',self.project))
            # run model
            self.fisheye_model(self.PATH_CAL, self.PATH_CALOUT)
            
        elif self.model=='default':
            # general cal output folder
            self.createFolders(os.path.join(self.PATH_CALOUT,'chessboardcorners'))
            self.createFolders(os.path.join(self.PATH_CALOUT,'undistort'))
            self.createFolders(os.path.join(self.PATH_CALOUT,'parameters'))
            self.createFolders(os.path.join(self.PATH_CALOUT,'chessboardcorners_fisheye',self.project))
            self.createFolders(os.path.join(self.PATH_CALOUT,'undistort_fisheye',self.project))
            self.createFolders(os.path.join(self.PATH_CALOUT,'parameters_fisheye',self.project))
            # run model
            self.default_model(self.PATH_CAL, self.PATH_CALOUT)
        
        return

