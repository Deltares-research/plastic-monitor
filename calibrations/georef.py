
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:01:28 2021

Run georeferencing and store homography matrix

@author: santinel
"""

#calibration caltech test

import numpy as np
import cv2 as cv
import os
import pandas as pd
import lenses

# TODO: Choosing the origin will improve the model results by highlighting significant digits
# TODO: Z values from GCP does not seem to be very accurate. Inquire or re-calibrate
# TODO: the H is considered always at z = 0m because (water) plane is considered as flat
# TODO: maybe useful to add origin rotation [compared to global reference] in the future
# TODO: print model errors

class georef:
    
    def __init__(self, 
                 PATH_IMAGES=[r'geooutput\TLC00009_00890_02300',
                   r'geooutput\TLC00010_00001_03608',
                   r'geooutput\TLC00011_00001_02836'], 
                 project='Bridge_20211020_cal2', 
                 PATH_PARAMS=r'caloutput\parameters_fisheye', 
                 PATH_GEOOUT = r'geooutput',
                 CSV_XYZs=r"c:\Users\santinel\Stichting Deltares\Robyn Gwee - videos\site2_bridge\20211021\point_bridge_20211021_UTM32749.csv", 
                 CSV_UVs=r"geooutput\point_bridge_gcps_final.csv"
                 ):
        # initialize paths
        self.PATH_IMAGES = PATH_IMAGES
        self.project = project
        self.PATH_PARAMS = PATH_PARAMS
        self.PATH_GEOOUT = PATH_GEOOUT
        # this is the one provided by Rizka, with UTM added.
        self.CSV_XYZs = CSV_XYZs
        # this is the one that pinpoint to the exact UV image coordinates
        self.CSV_UVs = CSV_UVs
        
        self.lens = lenses.lenses()

    def createFolders(self, pathname):
        # create folders
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        return
    
    
    # %% read camera origin
    def read_origin(self):
        '''
        read camera origin (approximately in the centre) from project name
    
        Parameters
        ----------
        location : string, optional
            project name, so that's picked up from right folder. 
            The default is 'Bridge_20211020_cal2'.
    
        Returns
        -------
        XYZc : array
            global coordinates [UTM, WGS84, ...] of camera centre/origin.
            Arbitrary values, chosen to make trailing digits more significant.
    
        '''
        
        XYZc=pd.read_csv(os.path.join(self.PATH_GEOOUT, self.project, 'origin.csv'))
        Xc = XYZc['Xc'][0]
        Yc = XYZc['Yc'][0]
        Zc = XYZc['Zc'][0]
        return Xc, Yc, Zc
    
    
    # %% working with cv Homography
    def find_homography(UV, XYZ, K, distortion=np.zeros((1,4)), z=0):
        '''Find homography based on ground control points
    
        Parameters
        ----------
        UV : np.ndarray
            Nx2 array of image coordinates of gcp's
        XYZ : np.ndarray
            Nx3 array of real-world coordinates of gcp's
        K : np.ndarray
            3x3 array containing camera matrix
        distortion : np.ndarray, optional
            1xP array with distortion coefficients with P = 4, 5 or 8
        z : float, optional
            Real-world elevation on which the image should be projected
    
        Returns
        -------
        np.ndarray
            3x3 homography matrix
    
        Notes
        -----
        Function uses the OpenCV image rectification workflow as described in
        https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html
        starting with solvePnP.
        '''
        UV = np.asarray(UV).astype(np.float32)
        XYZ = np.asarray(XYZ).astype(np.float32)
        K = np.asarray(K).astype(np.float32)
        
        # compute camera pose
        #    rvec, tvec = cv.solvePnP(XYZ, UV, K, distortion)[1:]
        _, rvec, tvec = cv.solvePnP(XYZ, UV, K, distortion)
        # convert rotation vector to rotation matrix
        R = cv.Rodrigues(rvec)[0]
        # assume height of projection plane
        R[:,2] = R[:,2] * z
        # add translation vector
        R[:,2] = R[:,2] + tvec.flatten()
        # compute homography
        H = np.linalg.inv(np.dot(K, R))
        # normalize homography
        H = H / H[-1,-1]
    
        return H
    
    
    # %% reading GCPs
    def readGCP(self):
        '''
        reading GCP data from CSVs
        '''
        # XYZ 
        df_XYZgcp = pd.read_csv(self.CSV_XYZs)
        # UV have been selected by running makegcp.py
        df_UVgcp = pd.read_csv(self.CSV_UVs)
        
        # make a df with common image filename, merging the two.
        dfgcp = pd.merge(df_XYZgcp, df_UVgcp, on=['img_filename'])
        
        # the one below to be used in case we can trust the elevation
        #XYZ = dfgcp.loc[:,['utm_x','utm_y','ele']].to_numpy()
        XYZ = np.c_[ dfgcp.loc[:,['utm_x','utm_y']].to_numpy(), np.zeros(np.shape(dfgcp)[0]) ]
        UV  = dfgcp.loc[:,['xgcp','ygcp']].to_numpy()
        # homography works on planes, points have to be undistorted 
        return XYZ, UV
    
    
    # %% compute Homography
    def computeHomography(self):
        '''
        Compute Homography after undistorting camera images, based on GCP insitu survey data
    
        Parameters
        ----------
        PATH_IMAGES : list
            List of paths where images are stores
        PATH_PARAMS : string
            path of lenses calibration parameters
        CSV_XYZs : string
            path to csv file of XYZ data of CSV
        CSV_UVs : string
            path to csv file of UV data of CSV
        project : string, optional
            project name. The default is 'Bridge_20211020_cal2'.
    
        Returns
        -------
        None.
    
        '''
        self.createFolders(os.path.join(self.PATH_GEOOUT, self.project))
        
        XYZ, UV = self.readGCP(self.CSV_XYZs, self.CSV_UVs)
        K, distortion = self.lens.readParams(self.PATH_PARAMS, self.project)
        
        # undistort UV
        UVd = self.lens.array2openCV(UV)
        UVu = self.lens.undistortPoints(UVd, K, distortion)
        UVu_np = self.lens.openCV2array(UVu)
        
        # centre to local coords
        Xc, Yc, Zc = self.read_origin()
        XYZ[:,0] = XYZ[:,0] - Xc
        XYZ[:,1] = XYZ[:,1] - Yc
        
        # find and save homography
        H = self.find_homography(UVu_np, XYZ, K, distortion, z=0) 
        path_H = os.path.join(self.PATH_GEOOUT, self.project, 'H.npy')
        with open(path_H, 'wb') as f:
            np.save(f, H)
        print('Homography computed and saved as '+ path_H)
        
        return 
    
    
    # %% utils 
    def get_pixel_coordinates(self, img):
        '''Get pixel coordinates given an image'''
        # get pixel coordinates
        U, V = np.meshgrid(range(img.shape[1]),
                           range(img.shape[0]))
        return U, V
    
    def rectify_image(self, img, H):
        '''Get projection of image pixels in real-world coordinates
           given an image and homography
        '''
        U, V = self.get_pixel_coordinates(img)
        X, Y = self.rectify_coordinates(U, V, H)
        
        return X, Y
    
    def rectify_coordinates(self, U, V, H):
        '''Get projection of image pixels in real-world coordinates
           given image coordinate matrices and  homography
    
        Parameters
        ----------
        U : np.ndarray
            NxM matrix containing u-coordinates
        V : np.ndarray
            NxM matrix containing v-coordinates
        H : np.ndarray
            3x3 homography matrix
    
        Returns
        -------
        np.ndarray
            NxM matrix containing real-world x-coordinates
        np.ndarray
            NxM matrix containing real-world y-coordinates
        '''
        UV = np.vstack((U.flatten(), V.flatten())).T
        # transform image using homography
        XY = cv.perspectiveTransform(np.asarray([UV]).astype(np.float32), H)[0]
        # reshape pixel coordinates back to image size
        X = XY[:,0].reshape(U.shape[:2])
        Y = XY[:,1].reshape(V.shape[:2])
        
        return X, Y
    
    
    def transformUV2XYZimage(self, img, H, K, distortion):
        '''
        Transform UV original ( distorted) image to global coordinates
        First performs undistortion from original UV to undistorted coordinates,
        then apply plane 2 plane transformation using H matrix.
        
        Returns
        -------
        X and Y in global reference system
    
        '''
        h,  w = img.shape[:2]
        scale_factor = 1.0 #DIM = [1920, 1080]
        DIM =(int(w*scale_factor), int(h*scale_factor))
        
        # init undistort
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3),
            K, DIM, cv.CV_16SC2)
        # undistort
        undist_image = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT)
        
        X, Y = self.rectify_image(undist_image, H)
        
        # centre to local coords
        Xc, Yc, Zc = self.read_origin()
        Xglob = np.float64(X) + Xc
        Yglob = np.float64(Y) + Yc
        
        return Xglob, Yglob
    
    
    def UV2XYZ(self, U, V, H, K, distortion):
        '''
        Geo Transformation including distortion
        Transform UV original ( distorted) coordinates to global coordinates
        First performs undistortion from original UV to undistorted coordinates,
        then apply plane 2 plane transformation using H matrix.
    
        Returns
        -------
        Transforms coordinates
    
        '''
        # preparing input
        UVd_cv = np.transpose(np.array([list(zip(U,V))]), axes=[1, 0, 2])
        
        # undistort UV
        UVu_cv = self.lens.undistortPoints(np.float32(UVd_cv), K, distortion)
        UVu = self.lens.openCV2array(UVu_cv)
        
        # use homography to get to XYZ
        X, Y = self.rectify_coordinates(UVu[:,0], UVu[:,1], H)
        
        # centre to local coords
        Xc, Yc, Zc = self.read_origin()
        Xglob = np.float64(X) + Xc
        Yglob = np.float64(Y) + Yc
        
        return Xglob, Yglob
    
    
    def XYZ2UV(self, Xglob, Yglob, H, K, distortion):
        '''
        Geo Transformation including distortion
        Transform XYZ global coordinates to original (distorted) pixel coordinates. 
        First performs plane 2 plane transformation using H matrix, then
        distort from undistorted to original (distorted) UV coordinates,
    
        Returns
        -------
        Transforms coordinates
    
        '''
        # preparing input
        # centre to local coords
        Xc, Yc, Zc = self.read_origin()
        X = np.float64(Xglob) - Xc
        Y = np.float64(Yglob) - Yc
        
        # use inverse of homography to get to UV from XYZ
        Hinv = np.linalg.pinv(H)
        Uu, Vu = self.rectify_coordinates(X, Y, Hinv)
        
        UVu_cv = np.transpose(np.array([list(zip(Uu,Vu))]), axes=[1, 0, 2])
        # distort UV
        UVd_cv = self.lens.distortPoints(np.float32(UVu_cv), K, distortion)
        UVd = self.lens.openCV2array(UVd_cv)
        
        return UVd[:,0], UVd[:,1]