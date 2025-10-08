#************************************************************ SUPPORTING LIBRARIES
import os;
import gc;
import numpy as np;
import time;
import sys;
import cv2;
import glob;
import pymeshlab;
import pydicom;
from tqdm import tqdm;
import SimpleITK as sitk
import matplotlib.pyplot as plt;
import trimesh;
import random;
import re;
import zipfile;
import shutil;
import pickle;
import pyvista as pv;
from scipy.spatial import KDTree;
from scipy.interpolate import RBFInterpolator;
from scipy.optimize import least_squares;
from sklearn.decomposition import PCA;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import Ridge;
from sklearn.cross_decomposition import CCA;
from sklearn.cross_decomposition import PLSRegression;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process.kernels import RBF;
from sklearn.mixture import GaussianMixture;
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler;
import matplotlib.pyplot as plt;
from matplotlib.font_manager import FontProperties;
import matplotlib.ticker as ticker;

import SupportingTools.SupportingTools as sp;
import VisualInterface.VisualInterface as vi;
from SMPLBody.SMPLBody import SmplBodyModel;
from SkeletonPredictor.SkeletonPredictor import SkeletonPredictor;
from SKELPredictor.SKELPredictor import SKELPredictor;
from PelvicBoneReconGUI.SystemDatabaseManager import SystemDatabaseManager;

import warnings
warnings.filterwarnings("ignore")

#************************************************************ SUPPORTING BUFFERS
mainFolder = "./";
dataFolder = mainFolder + "/Data";
rawDataFolder = dataFolder + "/Raw";
processedDataFolder = dataFolder + "/Processed";
postProcessedDataFolder = dataFolder + "/PostProcessed";
fineProcessedDataFolder = dataFolder + "/FineProcessed";
templateDataFolder = dataFolder + "/Template";
debugFolder = dataFolder + "/Debugs";
virtualCalibFolder = dataFolder + "/VirtualCalib";
virtualCalibDataFolder = virtualCalibFolder + "/CalibData";
virtualCalibResultFolder = virtualCalibFolder + "/CalibResults";
fusionDataFolder = virtualCalibFolder + "/FusionData";
fusionResultFolder = virtualCalibFolder + "/FusionResults";
viewer = vi.VisualInterface();
visionSeparation = 0.05;
capturingIndex = 0;
cameraIntrinsicMatrix = np.array([[1185.7,   0,   960,  0],
                                        [   0, 1185.7,  540,  0],
                                        [   0,   0,     1,    0],
                                        [   0,   0,     0,    1]    ]);
leftCameraName, rightCameraName, centerCameraName, upperCameraName, lowerCameraName = "leftCamera", "rightCamera", "centerCamera", "upperCamera", "lowerCamera";

#************************************************************ SUPPORTING FUNCTIONS
def isTopLeft(corners):
        first_point = corners[0][0]
        mean_x = np.mean(corners[:, 0, 0])
        mean_y = np.mean(corners[:, 0, 1])
        return first_point[0] < mean_x and first_point[1] < mean_y
def keyBoardCallBack(key):
    # Set the global variables
    global capturingIndex;
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

    # Capture position of the center camera
    if (key == 'p'):
        centerCamPosition = viewer.getCameraPosition("centerCamera");
        centerCamFocalPoint = viewer.getCameraFocalPoint("centerCamera");
        centerCamViewUp = viewer.getCameraViewUp("centerCamera");
        centerCamPosData = np.array([centerCamPosition, centerCamFocalPoint, centerCamViewUp]);
        sp.saveMatrixToCSVFile(debugFolder + "/centerCamPosData.csv", centerCamPosData)

    # Capture all images
    if (key == 'a'):
        print("Capturing image: ", capturingIndex);

        # Capture the image
        centerImage = viewer.captureCameraScreen("centerCamera", 1920, 1080);
        leftImage = viewer.captureCameraScreen("leftCamera", 1920, 1080);
        rightImage = viewer.captureCameraScreen("rightCamera", 1920, 1080);
        upperImage = viewer.captureCameraScreen("upperCamera", 1920, 1080);
        lowerImage = viewer.captureCameraScreen("lowerCamera", 1920, 1080);

        # Checking the chessboard detection in each image
        centerGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        leftGrayImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY);
        rightGrayImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY);
        upperGrayImage = cv2.cvtColor(upperImage, cv2.COLOR_BGR2GRAY);
        lowerGrayImage = cv2.cvtColor(lowerImage, cv2.COLOR_BGR2GRAY);

        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(centerGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(leftGrayImage, boardSize, None);
        ret3, corner3 = cv2.findChessboardCorners(rightGrayImage, boardSize, None);
        ret4, corner4 = cv2.findChessboardCorners(upperGrayImage, boardSize, None);
        ret5, corner5 = cv2.findChessboardCorners(lowerGrayImage, boardSize, None);

        if not (ret1 and ret2 and ret3 and ret4 and ret5):
            print("\t Cannot capture chessboard.");
            return;
        topLeft1 = isTopLeft(corners1);
        topLeft2 = isTopLeft(corners2);
        topLeft3 = isTopLeft(corner3);
        topLeft4 = isTopLeft(corner4);
        topLeft5 = isTopLeft(corner5);
        if not (topLeft1 and topLeft2 and topLeft3 and topLeft4 and topLeft5):
            print("\t The chessboards are not in the same direction.");
            return;

        # Save the image
        targetFolder = fusionDataFolder;
        leftImageFolder = targetFolder + "/LeftImages";
        rightImageFolder = targetFolder + "/RightImages";
        centerImageFolder = targetFolder + "/CenterImages";
        upperImageFolder = targetFolder + "/UpperImages";
        lowerImageFolder = targetFolder + "/LowerImages";

        sp.saveImage(centerImageFolder + f"/centerImage_{capturingIndex}.png", centerImage);
        sp.saveImage(leftImageFolder + f"/leftImage_{capturingIndex}.png", leftImage);
        sp.saveImage(rightImageFolder + f"/rightImage_{capturingIndex}.png", rightImage);
        sp.saveImage(upperImageFolder + f"/upperImage_{capturingIndex}.png", upperImage);
        sp.saveImage(lowerImageFolder + f"/lowerImage_{capturingIndex}.png", lowerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture center images
    if (key == 'b'):
        print("Capturing image: ", capturingIndex);

        # Checking image
        centerImage = viewer.captureCameraScreen("centerCamera", 1920, 1080);
        imageFolder = virtualCalibDataFolder + "/MonoCalibImages/CenterImages"; 
        imagePrefix = "centerImage";

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        if not (ret1): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Capture the image                     
        sp.saveImage(imageFolder + f"/{imagePrefix}_{capturingIndex}.png", centerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture left image
    if (key == 'c'):
        print("Capturing image: ", capturingIndex);

        # Checking image
        centerImage = viewer.captureCameraScreen("leftCamera", 1920, 1080);
        imageFolder = virtualCalibDataFolder + "/MonoCalibImages/LeftImages"; 
        imagePrefix = "leftImage";

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        if not (ret1): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Capture the image                     
        sp.saveImage(imageFolder + f"/{imagePrefix}_{capturingIndex}.png", centerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture right image
    if (key == 'd'):
        print("Capturing image: ", capturingIndex);

        # Checking image
        centerImage = viewer.captureCameraScreen("rightCamera", 1920, 1080);
        imageFolder = virtualCalibDataFolder + "/MonoCalibImages/RightImages"; 
        imagePrefix = "rightImage";

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        if not (ret1): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Capture the image                     
        sp.saveImage(imageFolder + f"/{imagePrefix}_{capturingIndex}.png", centerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture upper image
    if (key == 'e'):
        print("Capturing image: ", capturingIndex);

        # Checking image
        centerImage = viewer.captureCameraScreen("upperCamera", 1920, 1080);
        imageFolder = virtualCalibDataFolder + "/MonoCalibImages/UpperImages"; 
        imagePrefix = "upperImage";

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        if not (ret1): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Capture the image                     
        sp.saveImage(imageFolder + f"/{imagePrefix}_{capturingIndex}.png", centerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture lower image
    if (key == 'f'):
        print("Capturing image: ", capturingIndex);

        # Checking image
        centerImage = viewer.captureCameraScreen("lowerCamera", 1920, 1080);
        imageFolder = virtualCalibDataFolder + "/MonoCalibImages/LowerImages"; 
        imagePrefix = "lowerImage";

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        if not (ret1): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Capture the image                     
        sp.saveImage(imageFolder + f"/{imagePrefix}_{capturingIndex}.png", centerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;
    
    # Capture left right image
    if (key == 'g'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        leftImage = viewer.captureCameraScreen("leftCamera", 1920, 1080);
        rightImage = viewer.captureCameraScreen("rightCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        leftImageFolder = virtualCalibDataFolder + "/StereoCalibImages/LeftImages";
        rightImageFolder = virtualCalibDataFolder + "/StereoCalibImages/RightImages";
        sp.saveImage(leftImageFolder + f"/leftImage_{capturingIndex}.png", leftImage);
        sp.saveImage(rightImageFolder + f"/rightImage_{capturingIndex}.png", rightImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;
    
    # Capture upper lower images
    if (key == 'h'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        upperImage = viewer.captureCameraScreen("upperCamera", 1920, 1080);
        lowerImage = viewer.captureCameraScreen("lowerCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(upperImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(lowerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        upperImageFolder = virtualCalibDataFolder + "/StereoCalibImages/UpperImages";
        lowerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/LowerImages";
        sp.saveImage(upperImageFolder + f"/upperImage_{capturingIndex}.png", upperImage);
        sp.saveImage(lowerImageFolder + f"/lowerImage_{capturingIndex}.png", lowerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;
    
    # Capture center upper
    if (key == 'i'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        centerImage = viewer.captureCameraScreen("centerCamera", 1920, 1080);
        upperImage = viewer.captureCameraScreen("upperCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(upperImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        centerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterUpperImages/CenterImages";
        upperImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterUpperImages/UpperImages";
        sp.saveImage(centerImageFolder + f"/centerImage_{capturingIndex}.png", centerImage);
        sp.saveImage(upperImageFolder + f"/upperImage_{capturingIndex}.png", upperImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture center lower
    if (key == 'j'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        centerImage = viewer.captureCameraScreen("centerCamera", 1920, 1080);
        lowerImage = viewer.captureCameraScreen("lowerCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(lowerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        centerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterLowerImages/CenterImages";
        lowerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterLowerImages/LowerImages";
        sp.saveImage(centerImageFolder + f"/centerImage_{capturingIndex}.png", centerImage);
        sp.saveImage(lowerImageFolder + f"/lowerImage_{capturingIndex}.png", lowerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture center left
    if (key == 'k'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        centerImage = viewer.captureCameraScreen("centerCamera", 1920, 1080);
        leftImage = viewer.captureCameraScreen("leftCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        centerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterLeftImages/CenterImages";
        leftImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterLeftImages/LeftImages";
        sp.saveImage(centerImageFolder + f"/centerImage_{capturingIndex}.png", centerImage);
        sp.saveImage(leftImageFolder + f"/leftImage_{capturingIndex}.png", leftImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture center right
    if (key == 'l'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        centerImage = viewer.captureCameraScreen("centerCamera", 1920, 1080);
        rightImage = viewer.captureCameraScreen("rightCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(centerImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        centerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterRightImages/CenterImages";
        rightImageFolder = virtualCalibDataFolder + "/StereoCalibImages/CenterRightImages/RightImages";
        sp.saveImage(centerImageFolder + f"/centerImage_{capturingIndex}.png", centerImage);
        sp.saveImage(rightImageFolder + f"/rightImage_{capturingIndex}.png", rightImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture left upper
    if (key == 'm'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        leftImage = viewer.captureCameraScreen("leftCamera", 1920, 1080);
        upperImage = viewer.captureCameraScreen("upperCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(upperImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        leftImageFolder = virtualCalibDataFolder + "/StereoCalibImages/LeftUpperImages/LeftImages";
        upperImageFolder = virtualCalibDataFolder + "/StereoCalibImages/LeftUpperImages/UpperImages";
        sp.saveImage(leftImageFolder + f"/leftImage_{capturingIndex}.png", leftImage);
        sp.saveImage(upperImageFolder + f"/upperImage_{capturingIndex}.png", upperImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture left lower
    if (key == 'n'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        leftImage = viewer.captureCameraScreen("leftCamera", 1920, 1080);
        lowerImage = viewer.captureCameraScreen("lowerCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(lowerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        leftImageFolder = virtualCalibDataFolder + "/StereoCalibImages/LeftLowerImages/LeftImages";
        lowerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/LeftLowerImages/LowerImages";
        sp.saveImage(leftImageFolder + f"/leftImage_{capturingIndex}.png", leftImage);
        sp.saveImage(lowerImageFolder + f"/lowerImage_{capturingIndex}.png", lowerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture right upper
    if (key == 'o'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        rightImage = viewer.captureCameraScreen("rightCamera", 1920, 1080);
        upperImage = viewer.captureCameraScreen("upperCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(upperImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        rightImageFolder = virtualCalibDataFolder + "/StereoCalibImages/RightUpperImages/RightImages";
        upperImageFolder = virtualCalibDataFolder + "/StereoCalibImages/RightUpperImages/UpperImages";
        sp.saveImage(rightImageFolder + f"/rightImage_{capturingIndex}.png", rightImage);
        sp.saveImage(upperImageFolder + f"/upperImage_{capturingIndex}.png", upperImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Capture right lower
    if (key == 'p'):
        print("Capturing image: ", capturingIndex);
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001);

        # Capture the image
        rightImage = viewer.captureCameraScreen("rightCamera", 1920, 1080);
        lowerImage = viewer.captureCameraScreen("lowerCamera", 1920, 1080);

        # Checking images
        firstGrayImage = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY);
        secondGrayImage = cv2.cvtColor(lowerImage, cv2.COLOR_BGR2GRAY);
        boardSize = (9, 7);
        ret1, corners1 = cv2.findChessboardCorners(firstGrayImage, boardSize, None);
        ret2, corners2 = cv2.findChessboardCorners(secondGrayImage, boardSize, None);
        if not (ret1 and ret2): 
            print("\t Cannot detect chessboard.");
            return;
        corners1 = cv2.cornerSubPix(firstGrayImage, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(secondGrayImage, corners2, (11, 11), (-1, -1), criteria)
        if not (corners1[0][0][0] < np.mean(corners1[:, 0, 0]) and corners1[0][0][1] < np.mean(corners1[:, 0, 1]) and
                    corners2[0][0][0] < np.mean(corners2[:, 0, 0]) and corners2[0][0][1] < np.mean(corners2[:, 0, 1])): 
            print("\t Cannot find the same direction.");
            return;

        # Save the image
        rightImageFolder = virtualCalibDataFolder + "/StereoCalibImages/RightLowerImages/RightImages";
        lowerImageFolder = virtualCalibDataFolder + "/StereoCalibImages/RightLowerImages/LowerImages";
        sp.saveImage(rightImageFolder + f"/rightImage_{capturingIndex}.png", rightImage);
        sp.saveImage(lowerImageFolder + f"/lowerImage_{capturingIndex}.png", lowerImage);

        # Increase the capturing index
        capturingIndex = capturingIndex + 1;

    # Exit capturing
    if (key == 'q'):
        sys.exit();
def stereoCalibrateCameras(folderPath1, folderPath2, firstImagePrefix, secondImagePrefix, startIndex, endIndex, boardSize, squareSize, imageType="png"):
    # Define the criteria for corner refinement
    print("Define the criteria for corner refinement ...");
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialize object points based on the chessboard pattern
    print("Initialize object points based on the chessboard pattern ...");
    objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objp *= squareSize

    # Arrays to store object points and image points from all images
    print("Arrays to store object points and image points from all images ...");
    objpoints = []  # 3d points in real world space
    imgpoints1 = []  # 2d points in image plane for camera 1
    imgpoints2 = []  # 2d points in image plane for camera 2

    # Iterate through the image pairs
    print("Iterate through the image pairs ...");
    imageSize = (1920, 1080);
    for i in range(startIndex, endIndex + 1):
        img1_path = os.path.join(folderPath1, f"{firstImagePrefix}_{i}.{imageType}")
        img2_path = os.path.join(folderPath2, f"{secondImagePrefix}_{i}.{imageType}")
        img1 = cv2.imread(img1_path);
        height, width = img1.shape[:2];
        imageSize = (width, height);
        img2 = cv2.imread(img2_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        print("\t  Find the chessboard corners...");
        ret1, corners1 = cv2.findChessboardCorners(gray1, boardSize, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, boardSize, None)

        # If found, draw and display the corners
        print("\t  If found, draw and display the corners...")
        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            img1_with_corners = cv2.drawChessboardCorners(img1, boardSize, corners1, ret1)
            img2_with_corners = cv2.drawChessboardCorners(img2, boardSize, corners2, ret2)

            # Concatenate images horizontally
            print("\t  Concatenate images horizontally...")
            concatenated_img = np.hstack((img1_with_corners, img2_with_corners))
            # Calculate the new height to maintain the aspect ratio
            new_width = 1600
            original_height, original_width = concatenated_img.shape[:2]
            new_height = int((new_width / original_width) * original_height)
            concatenated_img_resized = cv2.resize(concatenated_img, (new_width, new_height))
            cv2.imshow("Chessboard Corners", concatenated_img_resized)
            key = cv2.waitKey(0)

            # If 'y' is pressed, add points to the lists
            print("\t If 'y' is pressed, add points to the lists ...")
            if key == ord('y'):
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)

    cv2.destroyAllWindows()

    # Calibrate the individual cameras
    print("Calibrate the individual cameras ...");
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)

    # Stereo calibration
    print("Stereo calibration ...");
    retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], 
                                                             criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC);
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T);

    # Output the extrinsic parameters and calibration error
    print("Output the extrinsic parameters and calibration error ...");
    return K1, D1, K2, D2, P1, P2;
def saveStereoCalibrationResults(filename, K1, D1, K2, D2, P1, P2):
    """
    Save camera parameters to an XML file using OpenCV.
    
    Parameters:
    - filename: str, path to the XML file.
    - K1, D1, K2, D2, P1, P2: numpy.ndarray, camera matrices to save.
    """
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

    # Write matrices
    fs.write("K1", K1)
    fs.write("D1", D1)
    fs.write("K2", K2)
    fs.write("D2", D2)
    fs.write("P1", P1)
    fs.write("P2", P2)

    # Release file
    fs.release()
def loadStereoCalibrationResults(filename):
    """
    Load camera parameters from an XML file using OpenCV.
    
    Parameters:
    - filename: str, path to the XML file.
    
    Returns:
    - K1, D1, K2, D2, P1, P2: numpy arrays containing the camera parameters.
    """
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    # Read matrices
    K1 = fs.getNode("K1").mat()
    D1 = fs.getNode("D1").mat()
    K2 = fs.getNode("K2").mat()
    D2 = fs.getNode("D2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()

    # Release file
    fs.release()

    return K1, D1, K2, D2, P1, P2
def computeProjectionMatrices(K1, K2, R, T):
    """
    Compute the projection matrices for a stereo camera setup.

    :param K1: Intrinsic matrix of the left camera (3x3)
    :param K2: Intrinsic matrix of the right camera (3x3)
    :param R: Rotation matrix from left to right camera (3x3)
    :param T: Translation vector from left to right camera (3,)
    :return: P1 (3x4), P2 (3x4) projection matrices
    """
    # Ensure T is a column vector (3,1)
    T = T.reshape(3, 1)

    # Left Camera Projection Matrix: P1 = K1 * [I | 0]
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Right Camera Projection Matrix: P2 = K2 * [R | T]
    P2 = K2 @ np.hstack((R, T))

    return P1, P2
def stereoReconstruct3D(firstPoints, secondPoints, P1, P2):
    """
    Perform 3D reconstruction from stereo images using multiple points.

    :param firstPoints: Nx2 array of points from the left camera.
    :param secondPoints: Nx2 array of points from the right camera.
    :param P1: 3x4 projection matrix of the left camera.
    :param P2: 3x4 projection matrix of the right camera.
    :return: Nx3 array of reconstructed 3D points.
    """
    # Ensure inputs are NumPy arrays with shape (2, N)
    firstPoints = np.array(firstPoints, dtype=np.float64).T  # Shape (2, N)
    secondPoints = np.array(secondPoints, dtype=np.float64).T  # Shape (2, N)

    # Perform triangulation (output is 4xN in homogeneous coordinates)
    points4D = cv2.triangulatePoints(P1, P2, firstPoints, secondPoints)

    # Convert from homogeneous to 3D Cartesian coordinates
    points3D = (points4D[:3] / points4D[3]).T  # Shape (N, 3)

    return points3D  # Return Nx3 array
def estimateEssentialMatrix(pts1, pts2, K):
    """ Compute the Essential Matrix using feature correspondences """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask
def recoverPoseAndTriangulate(pts1, pts2, E, K):
    """ Recover camera pose (R, t) and triangulate 3D points """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # Camera Projection Matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera at [I | 0]
    P2 = np.hstack((R, t))  # Second camera at [R | t]

    P1 = K @ P1  # Intrinsic * Extrinsic
    P2 = K @ P2

    # Triangulation
    pts1 = pts1.T
    pts2 = pts2.T
    points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # Convert from homogeneous coordinates
    points3D = points4D[:3] / points4D[3]
    return points3D.T
def reconstruct3DSFMBased(points1, points2, K):
    """
    Reconstructs 3D points from two 2D views by resolving motion ambiguity.

    Inputs:
        - points1: Nx2 array of 2D points in the first image.
        - points2: Nx2 array of 2D points in the second image.
        - K: 3x3 camera intrinsic matrix.

    Outputs:
        - best3DPoints: Nx3 array of reconstructed 3D points.
    """

    # Compute the Essential Matrix
    E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0)

    # Decompose E to find possible R and t
    U, _, Vt = np.linalg.svd(E)

    # Ensure U and Vt are proper rotation matrices
    W = np.array([[0, -1, 0], 
                  [1,  0, 0], 
                  [0,  0, 1]])

    # Possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Ensure rotation matrices are valid
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Four possible solutions for (R, t)
    solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

    def triangulate(R, t):
        """ Triangulates 3D points given a rotation R and translation t """
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for first camera
        P2 = K @ np.hstack((R, t.reshape(3, 1)))  # Projection matrix for second camera

        # Convert points to homogeneous format
        points1_h = points1.T
        points2_h = points2.T

        # Triangulate 3D points
        points4D_h = cv2.triangulatePoints(P1, P2, points1_h, points2_h)
        points3D = (points4D_h[:3] / points4D_h[3]).T  # Convert to 3D

        # Count valid points (Z > 0 in both cameras)
        valid_count = np.sum(points3D[:, 2] > 0)
        return valid_count, points3D

    # Select the best (R, t) by maximizing valid 3D points
    best_R, best_t, best3DPoints = max(
        [(R, t, triangulate(R, t)[1]) for R, t in solutions], 
        key=lambda item: np.sum(item[2][:, 2] > 0)
    )
    distances = np.linalg.norm(np.diff(best3DPoints, axis=0), axis=1)
    scaleFactor = np.mean(distances)
    if scaleFactor > 0:
        best3DPoints /= scaleFactor

    return best3DPoints
def reconstruct3DSFMBasedWithRealPoints(points1, points2, K, realPoints):
    """
    Reconstructs 3D points from two 2D views by resolving motion ambiguity and scaling with real-world points.

    Inputs:
        - points1: Nx2 array of 2D points in the first image.
        - points2: Nx2 array of 2D points in the second image.
        - K: 3x3 camera intrinsic matrix.
        - realPoints: Nx3 array of corresponding real-world 3D points for scale correction.

    Outputs:
        - best3DPoints: Nx3 array of reconstructed 3D points with corrected scale.
    """

    # Compute the Essential Matrix
    E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0)

    # Decompose E to find possible R and t
    U, _, Vt = np.linalg.svd(E)

    # Ensure U and Vt are proper rotation matrices
    W = np.array([[0, -1, 0], 
                  [1,  0, 0], 
                  [0,  0, 1]])

    # Possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    # Ensure valid rotation matrices
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Four possible (R, t) solutions
    solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

    def triangulate(R, t):
        """ Triangulates 3D points given a rotation R and translation t """
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for first camera
        P2 = K @ np.hstack((R, t.reshape(3, 1)))  # Projection matrix for second camera

        # Convert points to homogeneous format
        points1_h = points1.T
        points2_h = points2.T

        # Triangulate 3D points
        points4D_h = cv2.triangulatePoints(P1, P2, points1_h, points2_h)
        points3D = (points4D_h[:3] / points4D_h[3]).T  # Convert to 3D

        # Count valid points (Z > 0 in both cameras)
        valid_count = np.sum(points3D[:, 2] > 0)
        return valid_count, points3D

    # Select the best (R, t) by maximizing valid 3D points
    best_R, best_t, best3DPoints = max(
        [(R, t, triangulate(R, t)[1]) for R, t in solutions], 
        key=lambda item: np.sum(item[2][:, 2] > 0)
    )

    # Compute scale factor using realPoints
    if realPoints is not None and len(realPoints) == len(best3DPoints):
        # Compute mean distance between consecutive real points
        realDistances = np.linalg.norm(np.diff(realPoints, axis=0), axis=1)
        reconDistances = np.linalg.norm(np.diff(best3DPoints, axis=0), axis=1)

        scaleFactor = np.mean(realDistances) / np.mean(reconDistances) if np.mean(reconDistances) > 0 else 1.0
        best3DPoints *= scaleFactor  # Apply scale correction

    return best3DPoints
def extractCaseId(path):
    """
    Extracts 'case-XXXXX' from a file path string.

    Parameters:
    - path (str): The file path string.

    Returns:
    - str: The extracted case ID, e.g., 'case-100467'.
    """
    match = re.search(r'case-\d+', path)
    if match:
        return match.group(0)
    else:
        return None
def extractFolderFromZip(zipPath, targetFolderInZip, outputFolder):
    """
    Extracts all files from a specific folder inside a zip file and flattens them into a target output folder.

    Parameters:
    - zipPath (str): Path to the .zip file.
    - targetFolderInZip (str): Folder path inside the zip to extract from (e.g., 'case-100467/STANDARD_HEAD-NECK/').
    - outputFolder (str): Destination folder to extract files to.

    Notes:
    - Subdirectory structure from within the zip is ignored.
    - All extracted files are saved directly into the output folder.
    """
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        # Normalize path
        targetFolderInZip = targetFolderInZip.replace("\\", "/").rstrip("/") + "/"
        
        # Filter and extract files
        for member in zip_ref.namelist():
            if member.startswith(targetFolderInZip) and not member.endswith('/'):
                # Read the file and write it flat into the output folder
                filename = os.path.basename(member)
                targetPath = os.path.join(outputFolder, filename)
                
                # Create output directory if it doesn't exist
                os.makedirs(outputFolder, exist_ok=True)
                
                with zip_ref.open(member) as source, open(targetPath, 'wb') as target:
                    target.write(source.read())
def splitZipAndInternalPath(fullPath):
    # Match ZIP file path (ends with .zip) and separate internal path
    match = re.match(r'^(.*?\.zip)(?:[/\\]+(.*))?$', fullPath)
    if match:
        zipFilePath = match.group(1)
        internalPath = match.group(2) if match.group(2) else ''
        return zipFilePath, internalPath
    else:
        raise ValueError("Input path does not contain a valid .zip segment.")
def zipAndRemoveFolder(inputFolderPath, outputFolderPath):
    # Ensure paths are absolute
    inputFolderPath = os.path.abspath(inputFolderPath)
    outputFolderPath = os.path.abspath(outputFolderPath)

    # Extract folder name to use as zip file name
    folderName = os.path.basename(inputFolderPath)
    zipFilePath = os.path.join(outputFolderPath, f"{folderName}.zip")

    # Create output directory if it doesn't exist
    os.makedirs(outputFolderPath, exist_ok=True)

    # Create the zip file
    with zipfile.ZipFile(zipFilePath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(inputFolderPath):
            for file in files:
                filePath = os.path.join(root, file)
                # Write file to zip, preserving relative path inside folder
                arcname = os.path.relpath(filePath, inputFolderPath)
                zipf.write(filePath, arcname)

    # Remove the original folder and its contents
    shutil.rmtree(inputFolderPath)

    print(f"Zipped folder '{inputFolderPath}' to '{zipFilePath}' and removed the original folder.")

#*************************************************** PELVIS PREDICTION FROM PICKED FEATURES
# This section will predict the pelvis bone and muscle based on the feature points picked on the MRI or CT image. 
# The inputs:
#  + The pelvis shape data;
#  + The template pelvis bone;
#  + The template pelvis muscle;
# The outputs:
#  + The subject specific pelvis bone;
#  + The subject specific pelvis muscle;
# Processing procedure:
#  + Dataset preparation: 
#    -> The template pelvis bone, 
#    -> The template pelvis muscle, 
#    -> The pelvis shape data with feature points,
#    -> Other female pelvis CT meshes
#  + Dataset processing:
#    -> Scale the pelvis data into meter
#    -> Generate the pelvis shape from the CT pelvis meshes
#    -> Deform template pelvis shape to target pelvis shapes
#  + Pelvis statistical shape modeling 
#  + Pelvis bone regression based on feature points
#  + Cross-validation
#********************** TESTING FUNCTIONS
def testRigidTransform():
    # Initializing
    print("Initializing ...");
    templateFolder = r"H:\Data\Template";
    pelvisReconFolder = r"H:\Data\PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    debugFolder = r"H:\Data\PelvisBoneRecon\Debugs";

    # Reading template features
    print("Getting template information ...");
    tempShape = sp.readMesh(templateFolder + "/TemplatePelvisCoarseShape.ply");
    tempFeatures = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_picked_points.pp");

    # Reading target features
    print("Reading target information ...");
    targetShape = sp.readMesh(femalePelvisFolder + f"/{100131}-PelvisBoneShape.ply");
    targetFeatures = sp.read3DPointsFromPPFile(femalePelvisFolder + f"/{100131}-PelvisBoneMesh_picked_points.pp");

    # Estimate rigid transform
    print("Estimate rigid transform ...");
    svdTransform = sp.estimateAffineTransformCPD(tempFeatures, targetFeatures);
    defMesh = sp.transformMesh(tempShape, svdTransform);
    sp.saveMeshToPLY(debugFolder + "/defMesh.ply", defMesh);

    # Finished processing
    print("Finished processing.");
def testNonRigidICPRegistration():
    # Initializing
    print("Initializing ...");
    debugFolder = r"H:\Data\PelvisBoneRecon\Debugs";

    # Reading data
    print("Reading data ...");
    sourceMesh = sp.readMesh(templateDataFolder + "/TempPelvisBoneMesh.ply");
    sourceFeatures = sp.read3DPointsFromPPFile(templateDataFolder + "/TempPelvisBoneMesh_picked_points.pp");
    sourceFeatureIndices = sp.estimateNearestIndicesKDTreeBased(sourceFeatures, sourceMesh.vertices);
    targetMesh = sp.readMesh(debugFolder + "/100131-PelvisBoneMesh.ply");
    targetFeatures = sp.read3DPointsFromPPFile(debugFolder + "/100131-PelvisBoneMesh_picked_points.pp");

    # Rigid transform the mesh 
    print("Rigid transform the mesh ...");
    defMesh = sp.cloneMesh(sourceMesh);
    defFeatures = sourceFeatures.copy();
    rigidTransform = sp.estimateRigidSVDTransform(sourceFeatures, targetFeatures);
    defMesh = sp.transformMesh(defMesh, rigidTransform);
    defFeatures = sp.transform3DPoints(defFeatures, rigidTransform);

    # Deforming using Non-Rigid transform
    print("Deforming using Non rigid transform ...");
    nonRigidTransform = sp.estimateAffineTransformCPD(defFeatures, targetFeatures);
    defMesh = sp.transformMesh(defMesh, nonRigidTransform);
    defFeatures = sp.transform3DPoints(defFeatures, nonRigidTransform);

    # Deforming using Non-rigid ICP
    print("Deforming using Non-rigid ICP ...");
    defMeshVertices = trimesh.registration.nricp_amberg(
        source_mesh=defMesh,
        target_geometry=targetMesh,
        source_landmarks=sourceFeatureIndices,
        target_positions=targetFeatures
    )

    # Debugging
    print("Debugging ...");
    defMesh.vertices = defMeshVertices;
    sp.saveMeshToPLY(debugFolder + "/defMesh.ply", defMesh);

    # Finished processing
    print("Finished processing.");
def renameTheFeaturePoints():
    # Initializing
    print("Initializing ...");
    boneReconFolder = r"H:\Data\PelvisBoneRecon";
    femaleGeoFolder = r"H:\Data\PelvisBoneRecon\FemalePelvisGeometries";
    debugFolder = r"H:\Data\PelvisBoneRecon\Debugs";

    # Reading IDs
    print("Reading IDs ...");
    femaleIDs = sp.readListOfStrings(boneReconFolder + "/FemalePelvisIDs.txt");

    # Processing for template features
    print("Processing for template features ...");
    tempFeatures = sp.read3DPointsFromPPFile(r"H:\Data\Template/TempPelvisBoneMesh_picked_points.pp");
    sp.save3DPointsToPPFile(r"H:\Data\Template/TempPelvisBoneMesh_picked_points.pp", tempFeatures);
    
    # Processing for each subject
    print("Processing for each subject ...");
    for i, ID in enumerate(femaleIDs):
        # Debugging
        print("Processing subject: ", i, " with ID: ", ID);

        # Reading feature points
        features = sp.read3DPointsFromPPFile(femaleGeoFolder + f"/{ID}-PelvisBoneMesh_picked_points.pp");

        # Save feature back
        sp.save3DPointsToPPFile(debugFolder + f"/{ID}-PelvisBoneMesh_picked_points.pp", features);

    # Finished processing
    print("Finished processing.");
def fixNormalsForPelvisShape():
    # Initializing
    print("Initializing ...");
    def same_direction(vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        return dot_product > 0;
    def fix_mesh_normals_outward(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Fix mesh normals to ensure they point outward from the surface.
    
        Parameters:
            mesh (trimesh.Trimesh): Input mesh
    
        Returns:
            trimesh.Trimesh: Mesh with corrected face and vertex normals
        """
        # Ensure mesh is watertight before fixing normals
        if not mesh.is_watertight:
            mesh.fill_holes()

        # Reorient faces so normals point outward
        mesh = mesh.copy()
        mesh.fix_normals()  # Recomputes consistent winding order

        # Optionally re-estimate vertex normals from updated face normals
        mesh.vertex_normals = mesh.vertex_normals  # Force recomputation

        # Checking inward and outward again
        meshCentroid = sp.computeCentroidPoint(mesh.vertices);
        vertexNormals = mesh.vertex_normals.copy();
        for i, normal in enumerate(vertexNormals):
            outVector = mesh.vertices[i] - meshCentroid;
            vertexNormals[i] = outVector;
        mesh.vertex_normals = vertexNormals;

        # Return the mesh
        return mesh
    boneReconFolder = r"H:\Data\PelvisBoneRecon";
    femaleGeoFolder = r"H:\Data\PelvisBoneRecon\FemalePelvisGeometries";
    debugFolder = r"H:\Data\PelvisBoneRecon\Debugs";

    # Reading IDs
    print("Reading IDs ...");
    femaleIDs = sp.readListOfStrings(boneReconFolder + "/FemalePelvisIDs.txt");

    # Processing for each subject
    print("Processing for each subject ...");
    for i, ID in enumerate(femaleIDs):
        # Debugging
        print("Processing subject: ", i, " with ID: ", ID);

        # Reading feature points
        pelvisBoneShape = sp.readMesh(femaleGeoFolder + f"/{ID}-PelvisBoneShape.ply");

        # Fixing the normals
        pelvisBoneShape = fix_mesh_normals_outward(pelvisBoneShape);

        # Save mesh
        sp.saveMeshToPLY(debugFolder + f"/{ID}-PelvisBoneShape.ply", pelvisBoneShape);

    # Finished processing
    print("Finished processing.");
def findROIPelvisFeatureIndices():
    # Initialize
    print("Initializing ...");
    disk = "F:";
    templateFolder = disk + r"\Data\Template/PelvisBonesMuscles";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";

    # Reading full and ROI feature points
    print("Reading full and ROI feature points ...");
    fullFeatures = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points.off");
    leftIliumFeatures = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points_leftIlium.off");
    rightIliumFeatures = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points_rightIlium.off");
    sacrumFeatures = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points_sacrum.off");

    # Reading full pelvis mesh and ROI parts
    print("Reading full pelvis mesh and ROI parts ...");
    pelvisMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");
    pelvisLeftIliumMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_LeftIlium.ply");
    pelvisRightIliumMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_RightIlium.ply");
    pelvisSacrumMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_Sacrum.ply");
    pelvisLeftSacroiliacJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint.ply");
    pelvisRightSacroiliacJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint.ply");
    pelvisPubicJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_PubicJoint.ply");

    # Estimate the feature indices
    print("Estimate the feature indices ...");
    leftIliumFeatureIndices = sp.estimateNearestIndicesKDTreeBased(leftIliumFeatures, fullFeatures);
    rightIliumFeatureIndices = sp.estimateNearestIndicesKDTreeBased(rightIliumFeatures, fullFeatures);
    sacrumFeatureIndices = sp.estimateNearestIndicesKDTreeBased(sacrumFeatures, fullFeatures);
    sp.saveVectorXiToCSVFile(debugFolder + "/LeftIliumFeatureIndices.csv", leftIliumFeatureIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/RightIliumFeatureIndices.csv", rightIliumFeatureIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/SacrumFeatureIndices.csv", sacrumFeatureIndices);

    # Estimate ROI part indices
    print("Estimating ROI part indices ...");
    leftIliumVertexIndices = sp.estimateNearestIndicesKDTreeBased(pelvisLeftIliumMesh.vertices, pelvisMesh.vertices);
    rightIliumVertexIndices = sp.estimateNearestIndicesKDTreeBased(pelvisRightIliumMesh.vertices, pelvisMesh.vertices);
    sacrumVertexIndices = sp.estimateNearestIndicesKDTreeBased(pelvisSacrumMesh.vertices, pelvisMesh.vertices);
    leftSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(pelvisLeftSacroiliacJoint.vertices, pelvisMesh.vertices);
    rightSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(pelvisRightSacroiliacJoint.vertices, pelvisMesh.vertices);
    pubicVertexIndices = sp.estimateNearestIndicesKDTreeBased(pelvisPubicJoint.vertices, pelvisMesh.vertices);
    sp.saveVectorXiToCSVFile(debugFolder + "/LeftIliumVertexIndices.csv", leftIliumVertexIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/RightIliumVertexIndices.csv", rightIliumVertexIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/SacrumVertexIndices.csv", sacrumVertexIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/LeftSacroiliacJointVertexIndices.csv", leftSacroiliacJointVertexIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/RightSacroiliacJointVertexIndices.csv", rightSacroiliacJointVertexIndices);
    sp.saveVectorXiToCSVFile(debugFolder + "/PubicVertexIndices.csv", pubicVertexIndices);

    # Checking the feature indices
    print("Checking the feature indices ...");
    leftIliumFeatures_checking = fullFeatures[leftIliumFeatureIndices];
    rightIliumFeatures_checking = fullFeatures[rightIliumFeatureIndices];
    sacrumFeatures_checking = fullFeatures[sacrumFeatureIndices];
    sp.save3DPointsToOFFFile(debugFolder + "/fullFeatures_checking.off", fullFeatures);
    sp.save3DPointsToOFFFile(debugFolder + "/leftIliumFeatures_checking.off", leftIliumFeatures_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/rightIliumFeatures_checking.off", rightIliumFeatures_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/sacrumFeatures_checking.off", sacrumFeatures_checking);

    # Checking the ROI part indices
    print("Checking the ROI part indices ...");
    leftIliumVertices_checking = pelvisMesh.vertices[leftIliumVertexIndices];
    rightIliumVertices_checking = pelvisMesh.vertices[rightIliumVertexIndices];
    sacrumVertices_checking = pelvisMesh.vertices[sacrumVertexIndices];
    leftSacroiliacVertices_checking = pelvisMesh.vertices[leftSacroiliacJointVertexIndices];
    rightSacroiliacVertices_checking = pelvisMesh.vertices[rightSacroiliacJointVertexIndices];
    pubicVertices_checking = pelvisMesh.vertices[pubicVertexIndices];
    sp.save3DPointsToOFFFile(debugFolder + "/leftIliumVertices_checking.off", leftIliumVertices_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/rightIliumVertices_checking.off", rightIliumVertices_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/sacrumVertices_checking.off", sacrumVertices_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/leftSacroiliacVertices_checking.off", leftSacroiliacVertices_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/rightSacroiliacVertices_checking.off", rightSacroiliacVertices_checking);
    sp.save3DPointsToOFFFile(debugFolder + "/pubicVertices_checking.off", pubicVertices_checking);

    # Finished processing
    print("Finished processing.");
def testProjectionUsingNormals():
    # Initializing
    print("Initializing ...");
    debugFolder = r"G:\Data\PelvisBoneRecon\Debugs";

    # Reading the deformed mesh
    print("Reading the deformed mesh ...");
    defShape = sp.readMesh(debugFolder + "/125795-AlignedPelvisBoneShape.ply");
    targetShape = sp.readMesh(debugFolder + "/125795-PelvisBoneShape.ply");

    # Try to test projection using the normal nearest point
    print("Try to test the projection using normals ...");
    projectedMesh = sp.projectMeshOntoMesh(defShape, targetShape);

    # Save the computed results
    print("Saving the computed results ...");
    sp.saveMeshToPLY(debugFolder + "/projectedMesh.ply", projectedMesh);

    # Finished processing
    print("Finished processing.");
def testGeneratePelvisMeshFromPelvisShape_NoRigidAndNonRigidNormalization():
    # Initializing
    print("Initializing ...");
    def allVerticesInside(cage, target):
        """Check if all vertices of the target are inside the cage."""
        # Placeholder function: You should implement a proper point-in-mesh test
        return np.all(np.min(cage.vertices, axis=0) <= np.min(target.vertices, axis=0)) and \
               np.all(np.max(cage.vertices, axis=0) >= np.max(target.vertices, axis=0))
    templateFolder = r"G:\Data\Template";
    debugFolder = r"G:\Data\PelvisBoneRecon\Debugs";

    # Reading template pelvis mesh and shape
    print("Reading template pelvis mesh and shape ...");
    tempPelvisShape = sp.readMesh(templateFolder + "/TemplatePelvisCoarseShape.ply");
    tempPelvisMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");

    # Reading the target shape
    print("Target pelvis shape ...");
    targetPelvisShape = sp.readMesh(debugFolder + "/100131-AlignedPelvisBoneShape.ply");

    # Compute the Gaussian blenshape weights
    print("Compute the Gaussian blend shape weights ...");
    tempPelvisShapeWeights = sp.computeGaussianBlendShapeWeights(tempPelvisMesh.vertices, tempPelvisShape.vertices, 0.005);

    # Try to deform the pelvis shape
    print("Try to deform the pelvis shape ...");
    defPelvisMeshVertices = sp.deformMeshWithBlendShapeWeights(tempPelvisMesh.vertices, tempPelvisShape.vertices, 
                                                               targetPelvisShape.vertices, tempPelvisShapeWeights);
    defPelvisMesh = sp.cloneMesh(tempPelvisMesh);
    defPelvisMesh.vertices = defPelvisMeshVertices;

    # Save the computed mesh
    print("Save the deformed pelvis ...");
    sp.saveMeshToPLY(debugFolder + "/defPelvisMesh.ply", defPelvisMesh);

    # Finished processing
    print("Finished processing.");
def testGeneratePelvisMeshFromPelvisShape_WithRigidAndNonRigidNormalization_coupledGaussianDeformation():
    # Initializing
    print("Initializing ...");
    def allVerticesInside(cage, target):
        """Check if all vertices of the target are inside the cage."""
        # Placeholder function: You should implement a proper point-in-mesh test
        return np.all(np.min(cage.vertices, axis=0) <= np.min(target.vertices, axis=0)) and \
               np.all(np.max(cage.vertices, axis=0) >= np.max(target.vertices, axis=0))
    templateFolder = r"G:\Data\Template";
    debugFolder = r"G:\Data\PelvisBoneRecon\Debugs";

    # Reading template pelvis mesh and shape
    print("Reading template pelvis mesh and shape ...");
    tempPelvisShape = sp.readMesh(templateFolder + "/TemplatePelvisCoarseShape.ply");
    tempPelvisMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");

    # Reading the target shape
    print("Target pelvis shape ...");
    targetPelvisShape = sp.readMesh(debugFolder + "/100131-AlignedPelvisBoneShape.ply");    

    # Rigid and non rigid transform first
    print("Rigid and non-rigid transform first ...");
    defPelvisMesh = sp.cloneMesh(tempPelvisMesh);
    defPelvisShape = sp.cloneMesh(tempPelvisShape);
    svdTransform = sp.estimateRigidSVDTransform(defPelvisShape.vertices, targetPelvisShape.vertices);
    defPelvisMesh = sp.transformMesh(defPelvisMesh, svdTransform);
    defPelvisShape = sp.transformMesh(defPelvisShape, svdTransform);
    affineTransform = sp.estimateAffineTransformCPD(defPelvisShape.vertices, targetPelvisShape.vertices);
    defPelvisMesh = sp.transformMesh(defPelvisMesh, affineTransform);
    defPelvisShape = sp.transformMesh(defPelvisShape, affineTransform);

    # Compute the Gaussian blenshape weights
    print("Compute the Gaussian blend shape weights ...");
    tempPelvisShapeWeights = sp.computeGaussianBlendShapeWeights(defPelvisMesh.vertices, defPelvisShape.vertices, 0.005);

    # Try to deform the pelvis shape
    print("Try to deform the pelvis shape ...");
    defPelvisMeshVertices = sp.deformMeshWithBlendShapeWeights(defPelvisMesh.vertices, defPelvisShape.vertices, targetPelvisShape.vertices, tempPelvisShapeWeights);
    
    # Save the computed mesh
    print("Save the deformed pelvis ...");
    sp.saveMeshToPLY(debugFolder + "/defPelvisMesh.ply", defPelvisMesh);

    # Finished processing
    print("Finished processing.");
def testGeneratePelvisMeshFromPelvisShape_WithRigidAndNonRigidNormalization_coupledRadialBasicFunctionNonRigidICP():
    # Initializing
    print("Initializing ...");
    def allVerticesInside(cage, target):
        """Check if all vertices of the target are inside the cage."""
        # Placeholder function: You should implement a proper point-in-mesh test
        return np.all(np.min(cage.vertices, axis=0) <= np.min(target.vertices, axis=0)) and \
               np.all(np.max(cage.vertices, axis=0) >= np.max(target.vertices, axis=0))
    templateFolder = r"G:\Data\Template";
    debugFolder = r"G:\Data\PelvisBoneRecon\Debugs";

    # Reading template pelvis mesh and shape
    print("Reading template pelvis mesh and shape ...");
    tempPelvisShape = sp.readMesh(templateFolder + "/TemplatePelvisCoarseShape.ply");
    tempPelvisMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisFeatures = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_picked_points.pp");
    sp.save3DPointsToOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points.off", tempPelvisFeatures);

    # Reading the target shape
    print("Target pelvis shape ...");
    targetPelvisShape = sp.readMesh(debugFolder + "/100131-AlignedPelvisBoneShape.ply");
    targetPelvisMesh = sp.readMesh(debugFolder + "/100131-PelvisBoneMesh.ply");

    # Rigid and non rigid transform first
    print("Rigid and non-rigid transform first ...");
    defPelvisMesh = sp.cloneMesh(tempPelvisMesh);
    defPelvisShape = sp.cloneMesh(tempPelvisShape);
    svdTransform = sp.estimateRigidSVDTransform(defPelvisShape.vertices, targetPelvisShape.vertices);
    defPelvisMesh = sp.transformMesh(defPelvisMesh, svdTransform);
    defPelvisShape = sp.transformMesh(defPelvisShape, svdTransform);
    affineTransform = sp.estimateAffineTransformCPD(defPelvisShape.vertices, targetPelvisShape.vertices);
    defPelvisMesh = sp.transformMesh(defPelvisMesh, affineTransform);
    defPelvisShape = sp.transformMesh(defPelvisShape, affineTransform);

    # Deformation using the radial basic function
    print("Deformation using the radial basic function ...");
    displacements = targetPelvisShape.vertices - defPelvisShape.vertices;
    rbf = RBFInterpolator(defPelvisShape.vertices, displacements, kernel='thin_plate_spline');
    vertexDisplacement = rbf(defPelvisMesh.vertices);
    deformedVertices = defPelvisMesh.vertices + vertexDisplacement;
    defPelvisMesh.vertices = deformedVertices;

    # Try to deform the separate parts of the pelvis using non-rigid ICP
    print("Try to deform separate parts of the pelvis using non-rigid ICP ..");
    
    # Save the computed mesh
    print("Save the deformed pelvis ...");
    sp.saveMeshToPLY(debugFolder + "/defPelvisMesh.ply", defPelvisMesh);

    # Finished processing
    print("Finished processing.");
def testEstimateFaceIndicesOnOtherMeshes():
    # Initializing
    print("Initializing ...");
    def reconstructLandmarksFromBarycentric(mesh: trimesh.Trimesh, baryIndices: np.ndarray, baryCoords: np.ndarray) -> np.ndarray:
        """
        Reconstruct 3D points from barycentric coordinates and triangle indices.

        Args:
            mesh (trimesh.Trimesh): The source mesh.
            triIndices (np.ndarray): (n,) array of triangle indices.
            baryCoords (np.ndarray): (n, 3) array of barycentric coordinates.

        Returns:
            np.ndarray: (n, 3) array of reconstructed 3D points.
        """
        if baryIndices.shape[0] != baryCoords.shape[0]:
            raise ValueError("triIndices and baryCoords must have the same number of entries.")

        # Get the triangles from the mesh
        tris = mesh.triangles[baryIndices]  # shape (n, 3, 3)

        # Unpack vertices: V = w1*V0 + w2*V1 + w3*V2
        v0 = tris[:, 0, :]  # (n, 3)
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]

        w1 = baryCoords[:, 0][:, np.newaxis]
        w2 = baryCoords[:, 1][:, np.newaxis]
        w3 = baryCoords[:, 2][:, np.newaxis]

        points = w1 * v0 + w2 * v1 + w3 * v2  # (n, 3)
        return points
    def transferFaceIndicesToOtherMesh(sourceFaceIndices, sourceMesh, targetMesh):
        # Compute the vertex mapping
        sourceVertexIndicesOnTargetMesh = sp.estimateNearestIndicesKDTreeBased(sourceMesh.vertices, targetMesh.vertices);

        # Replace vertex indices on from mesh triangles
        sourceFaces = sourceMesh.faces[sourceFaceIndices];
        for i, face in enumerate(sourceFaces):
            face[0] = sourceVertexIndicesOnTargetMesh[face[0]];
            face[1] = sourceVertexIndicesOnTargetMesh[face[1]];
            face[2] = sourceVertexIndicesOnTargetMesh[face[2]];
            sourceFaces[i] = face;

        # Build the kdtree of the target faces
        targetFaceKDTree = KDTree(targetMesh.faces);

        # Query the point indices
        distanceBuffer, outFaceIndices = targetFaceKDTree.query(sourceFaces);

        # Return the outFaceIndices
        return outFaceIndices;
    secondDebugFolder = r"H:\Data\PelvisBoneRecon\Debugs_2";

    # Reading bary indices and coords
    print("Reading bary indices and coords ...");
    oldBaryIndices = sp.readIndicesFromCSVFile(secondDebugFolder + "/tempLeftSacroiliacJointFeatureBaryIndices.csv");
    oldBaryCoords = sp.readMatrixFromCSVFile(secondDebugFolder + "/tempLeftSacroiliacJointFeatureBaryCoords.csv");

    # Reading meshes
    print("Reading meshes ...");
    pelvisMesh = sp.readMesh(secondDebugFolder + "/tempPelvisMesh.ply");
    withoutJointPelvisMesh = sp.readMesh(secondDebugFolder + "/tempWithoutJointPelvisMesh.ply");

    # Testing the feature points on the old mesh
    print("Testing feature points on the old mesh ...");
    featurePoints = reconstructLandmarksFromBarycentric(withoutJointPelvisMesh, oldBaryIndices, oldBaryCoords);
    sp.save3DPointsToOFFFile(secondDebugFolder + "/featurePoints.off", featurePoints);

    # Testing transfering the bary indices
    print("Testing transfering the bary indices ...");
    newBaryIndices = transferFaceIndicesToOtherMesh(oldBaryIndices, withoutJointPelvisMesh, pelvisMesh);
    print("\t The shape of oldBaryIndices: ", oldBaryIndices.shape);
    print("\t The shape of newBaryIndices: ", newBaryIndices.shape);

    # Testing new bary indices
    print("Testing new bary indices ...");
    newBaryCoords = oldBaryCoords.copy();
    newFeaturePoints = reconstructLandmarksFromBarycentric(pelvisMesh, newBaryIndices, newBaryCoords);
    sp.save3DPointsToOFFFile(secondDebugFolder + "/newFeaturePoints.off", newFeaturePoints);

    # Finished processing
    print("Finished processing.");
def prepareTemplatePelvisFloorMuscles():
    # Initializing
    print("Initializing ...");

    # Reading mm floor muscles
    print("Reading mm floor muscles ...");

    # Finished processing
    print("Finished processing.");
def prepareFeaturePointIndices():
    # Intializing
    print("Initializing ...");
    disk = "G:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";

    # Reading data
    print("Reading data ...");
    featurePoints = sp.read3DPointsFromPPFile(debugFolder + "/TempPelvisBoneMuscles_AllFeaturePoints.pp");

    # Save feature points to off file
    print("Save feature points to off file ...");
    sp.save3DPointsToOFFFile(debugFolder + "/AllFeaturePoints.off", featurePoints);

    # Finished processing
    print("Finished processing.");
def estimateMappingFromNewToOldFeaturePicking():
    # Initializing
    print("Initializing ...");

    # Reading new and old picked feature points
    print("Reading new and old picked feature points ...");
    newFeaturePoints = sp.read3DPointsFromPPFile(debugFolder + "/TempPelvisBoneMesh_v2_picked_points.pp")
    oldFeaturePoints = sp.read3DPointsFromPPFile(debugFolder + "/TempPelvisBoneMesh_picked_points.pp")

    # Estimate the mapping old feature to old features
    print("Estimate the mapping from old to old feature picking ...");
    mappingIndices = sp.estimateNearestIndicesKDTreeBased(oldFeaturePoints, newFeaturePoints);

    # Save the mapping indices
    print("Save the mapping indices ...");
    sp.saveVectorXiToCSVFile(debugFolder + "/FeatureMappingIndices.csv", mappingIndices);

    # Finished processing
    print("Finished processing.");
def loadSaveNewOldPelvisFeaturePoints():
    # Initializing
    print("Initializing ...");
    templatePelvisMuscleFolder = r"I:\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    debugFolder = r"I:\SpinalPelvisPred\Data\PelvisBoneRecon\Debugs";

    # Reading old feature points
    print("Reading old feature points ...");
    oldFeaturePoints = sp.read3DPointsFromPPFile(templatePelvisMuscleFolder + "/TempPelvisBoneMesh_picked_points.pp");
    newFeaturePoints = sp.read3DPointsFromPPFile(templatePelvisMuscleFolder + "/TempPelvisBoneMesh_picked_points_v2.pp");
    pelvisBone = sp.readMeshFromPLY(templatePelvisMuscleFolder + "/TempPelvisBoneMesh.ply");

    # Debugging
    print("Debugging ...");
    sp.saveMeshToPLY(debugFolder + "/TempPelvisBoneMesh.ply", pelvisBone);
    sp.save3DPointsToOFFFile(debugFolder + "/TempPelvisBoneMesh_picked_points.off", oldFeaturePoints);
    sp.save3DPointsToOFFFile(debugFolder + "/TempPelvisBoneMesh_picked_points_v2.off", newFeaturePoints);

    # Finished procesing
    print("Finished processing.");
def testingIndicesMappingForPelvisFeatures():
    # Initializing
    print("Initializing ...");
    debugFolder = r"I:\SpinalPelvisPred\Data\PelvisBoneRecon\Debugs";

    # Reading old feature points
    print("Reading old feature points ...");
    oldFeaturePoints = sp.read3DPointsFromPPFile(debugFolder + "/7-PelvisBoneMesh_picked_points.pp");

    # Reading mapping indices
    print("Reading mapping indices ...");
    mappingIndices = sp.readIndicesFromCSVFile(debugFolder + "/NewMexicoTo1KPelvisFeatureMappingIndices.csv");

    # Forming new feature points
    print("Forming new feature points ...");
    newFeaturePoints = oldFeaturePoints[mappingIndices];

    # Reading template pelvis mesh and features
    print("Reading template pelvis mesh and features ...");
    templatePelvisMesh = sp.readMesh(debugFolder + "/TempPelvisBoneMesh.ply");
    templatePelvisFeatures = sp.read3DPointsFromPPFile(debugFolder + "/TempPelvisBoneMesh_picked_points.pp");

    # Debugging
    print("Debugging ...");
    sp.save3DPointsToOFFFile(debugFolder + "/NewFeaturePoints.off", newFeaturePoints);
    sp.save3DPointsToOFFFile(debugFolder + "/OldFeaturePoints.off", oldFeaturePoints);
    sp.save3DPointsToOFFFile(debugFolder + "/TemplatePelvisFeatures.off", templatePelvisFeatures);

    # Finished processing
    print("Finished processing.");
def fix1KPelvisFeaturePoints():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("\t Please write the command like: [ProgramName] [StartSubjectIndex] [EndSubjectIndex]");
        return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);
    processingDisk = "I:/SpinalPelvisPred";
    templateFolder = processingDisk + "/Data/Template";
    pelvisBoneMuscleTempFolder = templateFolder + "/PelvisBonesMuscles";
    pelvisReconFolder = processingDisk +  "/Data/PelvisBoneRecon";
    targetPelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries/1KPelvisData";
    debugFolder = processingDisk + "/Data/PelvisBoneRecon/Debugs";

    # Reading feature mapping indices
    print("Reading feature mapping indices ...");
    mappingIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/NewMexicoTo1KPelvisFeatureMappingIndices.csv");
    print("\t The number of indices: ", mappingIndices.shape[0]);

    # Reading 1k pelvis IDs
    print("Reading 1K pelvis IDs ...");
    subjectIDs = sp.readListOfStrings(pelvisReconFolder + "/FemalePelvisGeometries/1KPelvisDataFemalePelvisIDs.txt");
    print("\t The number of subject IDs: ", len(subjectIDs));

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(startIndex, endIndex + 1):        
        # Debugging
        sID = subjectIDs[i];
        print("\t Processing subject: ", i, " with ID: ", sID);

        # Reading the feature points
        featurePointsFilePath = targetPelvisFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp";
        oldFeatures = sp.read3DPointsFromPPFile(featurePointsFilePath);

        # Fixing feature points
        fixedFeatures = oldFeatures[mappingIndices];

        # Save the fixed feature points
        sp.save3DPointsToPPFile(debugFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp", fixedFeatures);

    # Finished processing
    print("Finished processing.");
def testEstimatingROIFeatureIndices():
    # Initializing
    print("Initializing ...");
    debugFolder = r"I:\SpinalPelvisPred\Data\Debugs";

    # Reading full pelvis tructure and roi pelvis tructure
    print("Reading full pelvis structure and roi pelvis structure ...");
    fullPelvisStructure = sp.readMesh(debugFolder + "/TempPelvisBoneMuscles.ply");
    roiPelvisStructure = sp.readMesh(debugFolder + "/TempPelvisBoneMesh_LeftIlium.ply");

    # Finished processing
    print("Finished processing.");

#********************************************************** SUPPORTING FUNCTIONS

#********************************************************** CHECKING FUNCTIONS
def checkPelvisFeaturePointsFor1KPelvisData():
    # Initializing
    print("Initializing ...");
    def extractIDFromFileName(fileName, suffix):
        """
        Extract numeric ID from a file name.
        
        Parameters:
        - fileName (str): The file name (e.g., "case-100467-something.ext" or "100467-PelvisBoneMesh.ply")
        - suffix (str): Optional suffix to remove from the end
        
        Returns:
        - str: The extracted numeric ID (e.g., "100467")
        """
        # Get just the filename without any path components
        baseName = os.path.basename(fileName)
        
        # Remove the suffix if provided
        if suffix and baseName.endswith(suffix):
            baseName = baseName[:-len(suffix)]
        
        # Remove file extension if no suffix was provided
        if not suffix:
            baseName = os.path.splitext(baseName)[0]
        
        # Extract numeric ID using regex
        # This handles patterns like "case-100467", "100467-something", or just "100467"
        match = re.search(r'\b(\d+)\b', baseName)
        
        if match:
            return match.group(1)
        else:
            return None
    pelvisReconFolder = r"I:\SpinalPelvisPred\Data\PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    kPelvisFolder = femalePelvisFolder + "/1KPelvisData";

    # Prepare IDs for the 1K pelvis data
    print("Prepare IDs for the 1K pelvis data ...");
    ## List all pp file inside the folder
    ppFiles = sp.listAllFilesWithExtensionInsideAFolder(kPelvisFolder, ".pp");
    ## Extract the IDs from the file names, the file name has the format: {ID}-PelvisBoneMesh_picked_points.pp
    subjectIDs = [extractIDFromFileName(fileName, "-PelvisBoneMesh_picked_points.pp") for fileName in ppFiles];
    ## Save the IDs to the debug folder
    sp.saveListOfStrings(femalePelvisFolder + "/1KPelvisDataFemalePelvisIDs.txt", subjectIDs);

    # Initialize visual interface for checking visualization
    print("Initialize visual interface for checking visualization ...");
    viewer.initializeAutoSingleRenderer();
    viewer.setWindowSize(1600, 900);
    viewer.setTrackballCameraWindowInteractor();
    viewer.setBackgroundColorByName("white");
    
    # Checking for each subject
    print("Checking for each subject ...");
    for i, sID in enumerate(subjectIDs):
        # Debugging 
        print("Processing subject: ", i, " with ID: ", sID, end="", flush=True);

        # Reading pelvis bone mesh and shape
        pelvisBoneMesh = sp.readMesh(kPelvisFolder + f"/{sID}-PelvisBoneMesh.ply");
        pelvisBoneShape = sp.readMesh(kPelvisFolder + f"/{sID}-PelvisBoneShape.ply");

        # Reading features
        pickPointFilePath = kPelvisFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp";
        features = sp.read3DPointsFromPPFile(pickPointFilePath);

        # Visualize the shape mesh with feature points
        viewer.addMesh("PelvisBoneShape", pelvisBoneShape, "blue");
        viewer.addMesh("PelvisBoneMesh", pelvisBoneMesh, "red");
        for j, feature in enumerate(features):
            viewer.addColorSphereMesh(f"FeaturePoint_{j}", feature, 0.005);
        viewer.setMeshOpacity("PelvisBoneShape", 0.5);
        viewer.resetMainCamera();
        viewer.render();

        # Start window interactor
        viewer.startWindowInteractor();

        # Remove all rendered objects
        viewer.removeAllRenderedObjects();

        # Checking len
        if (len(features) == 53): print("-> OK"); 
        else: print("-> NO OK");

    # Finished processing
    print("Finished processing.");

#********************************************************** PROCESSING FUNCTIONS
#************************** DATA PROCESSING FUNCTIONS
def reOrganise1KPelvisMeshes():
    # Initialize
    print("Initializing ...");
    targetFolder = r"G:\Data\Others\1KPelvis\3DMeshes";
    reOrganizedFolder = r"G:\Data\Others\1KPelvis";

    # List folder names
    print("Listing folder names ...");
    folderNames = sp.listAllFolderNames(targetFolder);

    # Process for each folder name
    print("Process for each folder name ...");
    subjectID = 0;
    for folderName in folderNames:
        # Debugging
        print("\t Processing folder: ", folderName);

        # Reading mesh
        pelvisBoneMesh = sp.readMesh(f"{targetFolder}/{folderName}/TempSegmentation.stl");

        # Save the mesh
        sp.saveMeshToPLY(f"{reOrganizedFolder}/{subjectID}-PelvisBoneMesh.ply", pelvisBoneMesh);
        subjectID += 1;

    # Finished processing
    print("Finished processing.");
def formPelvisFrom1KPelvisMeshes():
    # Initializing
    print("Initializing ...");
    pelvisMeshFolder = r"H:\Data\Others\1KPelvis\1KPelvisMeshes";
    mergedPelvisMeshFolder = r"H:\Data\Others\1KPelvis\1KPelvisMeshes_Merged";
    numOfSubjects = 1106;

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(numOfSubjects):
        # Debugging
        print("\t Processing subject: ", i);

        # Reading meshes
        meshParts = [];
        for j in range(1, 4):
            meshParts.append(sp.readMesh(f"{pelvisMeshFolder}/Model_{i}/TempSegmentation_{j}.stl"));

        # Merge the mesh part
        pelvisMesh = sp.mergeMeshes(meshParts);

        # Save mesh
        sp.saveMeshToPLY(f"{mergedPelvisMeshFolder}/{i}-PelvisBoneMesh.ply", pelvisMesh);

    # Finished processing
    print("Finished processing.");
def scale1KPelvisToMeter():
    # Initializing
    print("Initializing ...");
    inputFolder = r"H:\Data\Others\1KPelvis\1KPelvisMeshes_Merged";
    outputFolder = r"H:\Data\Others\1KPelvis\1KPelvisMeshes_Scaled";
    numOfModels = 1106;

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(numOfModels):
        # Debugging
        print("\t Processing subject: ", i);

        # Reading model
        pelvisMesh = sp.readMesh(f"{inputFolder}/{i}-PelvisBoneMesh.ply");

        # Scale the model
        scaledMesh = sp.scaleMesh(pelvisMesh, 0.001);

        # Save mesh to files
        sp.saveMeshToPLY(f"{outputFolder}/{i}-PelvisBoneMesh.ply", scaledMesh);

    # Finished processing
    print("Finished processing.");
def prepareTemplatePelvis():
    # Initializing
    print("Initializing ...");

    # Reading mesh and feautre
    print("Reading mesh and features ...");
    tempPelvisBoneMesh = sp.readMesh(templateDataFolder + "/PelvisBoneMesh.off");
    tempPelvisBoneFeatures = sp.read3DPointsFromPPFile(templateDataFolder + "/PelvisBoneMesh_picked_points.pp");

    # Scale the mesh to meter
    print("Scale the mesh to meter ...");
    tempPelvisBoneMesh = sp.scaleMesh(tempPelvisBoneMesh, 0.001);
    tempPelvisBoneFeatures = sp.scale3DPoints(tempPelvisBoneFeatures, 0.001);

    # Save the scaled value
    print("Save the scaled value ...");
    sp.saveMeshToPLY(debugFolder + "/PelvisBoneMesh.off", tempPelvisBoneMesh);
    sp.save3DPointsToPPFile(debugFolder + "/PelvisBoneMesh_picked_points.pp", tempPelvisBoneFeatures);

    # Finished processing
    print("Finished processing.");
def generateTemplatePelvisShape():
    # Initialize
    print("Initializing ...");
    
    # Reading mesh
    print("Reading mesh ...");
    pelvis = sp.readMesh(templateDataFolder + "/PelvisBoneMesh.ply");

    # Estimating the pelvis shape 
    print("Estimating pelvis shape ...");
    pelvisShape = sp.estimatePelvisShape(pelvis, 4, 0.001);
    sp.saveMeshToPLY(debugFolder + "/pelvisShape.ply", pelvisShape);

    # Finished processing
    print("Finished processing.");
def generateROIPelvisPartIndices():
    # Initializing
    print("Initializing ...");

    # Finished processing
    print("Finished processing.");
def generatePelvisShapes():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("\t Please input the command as the following: [ProgramName] [StartIndex] [EndIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);
    disk = "H:";
    targetFolder = disk + r"\Data\PelvisBoneRecon\FemalePelvisGeometries";
    outFolder = disk + r"\Data\PelvisBoneRecon\Debugs";

    # Read IDs list
    print("Read IDs ...");
    femaleIDs = sp.readListOfStrings(disk + r"\Data\PelvisBoneRecon/FemalePelvisIDs.txt");

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(startIndex, endIndex + 1):
        # Debugging
        subjectID = femaleIDs[i];
        print("\t Processing subject: ", i, "->", subjectID);

        # Reading pelvis bone mesh
        pelvisBoneMesh = sp.readMesh(targetFolder + f"/{subjectID}-PelvisBoneMesh.ply");
        sp.saveMeshToPLY(debugFolder + "/pelvisBoneMesh.ply", pelvisBoneMesh);

        # Estimate the shape
        pelvisBoneShape = sp.estimatePelvisShape(pelvisBoneMesh);

        # Save the pelvis shape
        sp.saveMeshToPLY(f"{outFolder}/{subjectID}-PelvisBoneShape.ply", pelvisBoneShape);

    # Finished processing
    print("Finished processing.");
def getFeaturePointsOfFemalePelvisBoneShape():
    # Initializing
    print("Initializing ...");
    pelvisBoneReconFolder = "H:/Data/PelvisBoneRecon";
    femalePelvisFolder = pelvisBoneReconFolder + "/FemalePelvisGeometries";
    postReconFolder = r"H:\Data\PostProcessed";
    debugFolder = pelvisBoneReconFolder + "/Debugs";

    # Get female pelvis IDs
    print("Getting female pelvis IDs ...");
    pelvisIDs = sp.readListOfStrings(pelvisBoneReconFolder + "/FemalePelvisIDs.txt");
    numOfSubjects = len(pelvisIDs);    
    
    # Getting pelvis feature for all subjects
    print("Getting pelvis features for all subjects ...");
    for i in range(numOfSubjects):
        # Debugging
        pelvisID = pelvisIDs[i];
        print("Processing subject: ", pelvisIDs[i]);

        # Reading features
        originSpinoPelvicFeatures = sp.read3DPointsFromPPFile(f"{postReconFolder}/{pelvisID}-SpinopelvicBoneMesh_picked_points.pp");
        transFeatures = sp.read3DPointsFromPPFile(femalePelvisFolder + f"/{pelvisID}-PelvisBoneMesh_picked_points.pp");
        
        # Estimate translation matrix
        transMatrix = sp.estimateRigidSVDTransform(originSpinoPelvicFeatures[0:22], transFeatures);

        # Transform original features
        transSpinoPelvicFeatures = sp.transform3DPoints(originSpinoPelvicFeatures, transMatrix);
        transPelvicFeatures = transSpinoPelvicFeatures[0:23];

        # Save to the feature folder
        sp.save3DPointsToPPFile(f"{debugFolder}/{pelvisID}-PelvisBoneMesh_picked_points.pp", transPelvicFeatures)

    # Finished processing
    print("Finished processing.");
def normalizeFemalePelvisBoneShapes_centroidToOrigin():
    # Initializing
    print("Initializing ...");
    pelvisReconFolder = r"H:\Data\PelvisBoneRecon";
    pelvisGeoFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    debugFolder = pelvisReconFolder + "/Debugs";

    # Reading female pelvis ids
    print("Getting female pelvis ids ...");
    femalePelvisIDs = sp.readListOfStrings(pelvisReconFolder + "/FemalePelvisIDs.txt");
    numOfSujects = len(femalePelvisIDs);

    # Register all pelvis to the original coordinate system
    print("Register all pelvis to the original coordinate system ...");
    for i, sID in enumerate(femalePelvisIDs):
        # Debugging
        print("\t Processing subject: ", sID);

        # Reading features
        features = sp.read3DPointsFromPPFile(pelvisGeoFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp");

        # Compute feature centroid
        featureCentroid = sp.computeCentroidPoint(features);
        orginCentroid = np.array([0, 0, 0]);

        # Estimate transform matrix
        transMatrix = sp.estimateTranslationMatrixFromSourceToTargetPoint(featureCentroid, orginCentroid);

        # Reading pelvis bone mesh and transform to the origin
        originPelvisMesh = sp.readMesh(pelvisGeoFolder + f"/{sID}-PelvisBoneMesh.ply");
        originPelvisShape = sp.readMesh(pelvisGeoFolder + f"/{sID}-PelvisBoneShape.ply")

        # Transform the meshes and feature points
        transPelvisMesh = sp.transformMesh(originPelvisMesh, transMatrix);
        transPelvisShape = sp.transformMesh(originPelvisShape, transMatrix);
        transFeatures = sp.transform3DPoints(features, transMatrix);

        # Save mesh to debug folder
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PelvisBoneMesh.ply", transPelvisMesh);
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PelvisBoneShape.ply", transPelvisShape);
        sp.save3DPointsToPPFile(debugFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp", transFeatures)

    # Finished processing
    print("Finished processing.");
def normalizeFemalePelvisBoneShapes_toFirstFeatures():
    # Initializing
    print("Initializing ...");
    pelvisReconFolder = r"H:\Data\PelvisBoneRecon";
    pelvisGeoFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    debugFolder = pelvisReconFolder + "/Debugs";

    # Reading female pelvis ids
    print("Getting female pelvis ids ...");
    femalePelvisIDs = sp.readListOfStrings(pelvisReconFolder + "/FemalePelvisIDs.txt");
    numOfSujects = len(femalePelvisIDs);

    # Reading first feature set
    print("Reading first feature set ...");
    firstFeatures = sp.read3DPointsFromPPFile(pelvisGeoFolder + f"/{femalePelvisIDs[0]}-PelvisBoneMesh_picked_points.pp");

    # Register all pelvis to the original coordinate system
    print("Register all pelvis to the original coordinate system ...");
    for i, sID in enumerate(femalePelvisIDs):
        # Debugging
        print("\t Processing subject: ", sID);

        # Reading features
        originFeatures = sp.read3DPointsFromPPFile(pelvisGeoFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp");        
        originPelvisMesh = sp.readMesh(pelvisGeoFolder + f"/{sID}-PelvisBoneMesh.ply");
        originPelvisShape = sp.readMesh(pelvisGeoFolder + f"/{sID}-PelvisBoneShape.ply")

        # Estimate transform matrix
        transMatrix = sp.estimateRigidSVDTransform(originFeatures, firstFeatures);

        # Transform the meshes and feature points
        transPelvisMesh = sp.transformMesh(originPelvisMesh, transMatrix);
        transPelvisShape = sp.transformMesh(originPelvisShape, transMatrix);
        transFeatures = sp.transform3DPoints(originFeatures, transMatrix);

        # Save mesh to debug folder
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PelvisBoneMesh.ply", transPelvisMesh);
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PelvisBoneShape.ply", transPelvisShape);
        sp.save3DPointsToPPFile(debugFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp", transFeatures)

    # Finished processing
    print("Finished processing.");
def normalizePelvisBoneShapes_centroidToOrigin():
    # Initializing
    print("Initializing ...");
    pelvisReconFolder = r"H:\Data\PelvisBoneRecon";
    pelvisGeoFolder = pelvisReconFolder + "/PelvisGeometries";
    debugFolder = pelvisReconFolder + "/Debugs";

    # Reading subject IDs
    print("Reading subject IDs ...");
    subjectIDs = sp.readListOfStrings(pelvisReconFolder + "/PelvisIDs.txt");

    # Processing for each subject
    print("Processing for each subject ...");
    for i, sID in enumerate(subjectIDs):
        # Debugging
        print("\t Processing subject: ", i, " with ID: ", sID);

        # Reading pelvis
        originPelvis = sp.readMesh(pelvisGeoFolder + f"/{sID}-PelvisBoneMesh.ply");
        originPelvisShape = sp.readMesh(pelvisGeoFolder + f"/{sID}-PelvisBoneShape.ply");

        # Compute centroid points
        originPelvisCentroid = sp.computeCentroidPoint(originPelvis.vertices);

        # Estimate transform matrix
        transMatrix = sp.estimateTranslationMatrixFromSourceToTargetPoint(originPelvisCentroid, np.array([0, 0, 0]));

        # Transform pelvis
        transPelvis = sp.transformMesh(originPelvis, transMatrix);
        transPelvisShape = sp.transformMesh(originPelvisShape, transMatrix);

        # Save the pelvis
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PelvisBoneMesh.ply", transPelvis);
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PelvisBoneShape.ply", transPelvisShape);
    
    # Finished processing
    print("Finished processing.");
def deformTemplatePelvisToTargetPelvis():
    # Initialize
    print("Initializing ...");
    def computeBarycentricLandmarks(mesh: trimesh.Trimesh, landmarkPoints: np.ndarray):
        """
        Given 3D landmark points and a mesh, compute the triangle indices and
        barycentric coordinates for use in trimesh.registration.nricp_amberg.

        Args:
            mesh (trimesh.Trimesh): The mesh where landmarks will be localized.
            landmarkPoints (np.ndarray): (n, 3) array of 3D landmark points.

        Returns:
            tuple:
                triIndices (np.ndarray): (n,) int array of triangle indices.
                baryCoords (np.ndarray): (n, 3) float array of barycentric coordinates.
        """
        closest = trimesh.proximity.closest_point(mesh, landmarkPoints)
        triIndices = closest[2]      # (n,) array of triangle indices
        
        baryCoords = []
        for point, triIndex in zip(landmarkPoints, triIndices):
            triVerts = mesh.triangles[triIndex]  # shape (3, 3)
            v0, v1, v2 = triVerts

            # Compute barycentric coordinates
            T = np.column_stack([v1 - v0, v2 - v0])
            v = point - v0
            A = np.dot(T.T, T)
            b = np.dot(T.T, v)

            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.zeros(2)

            w1 = 1.0 - x[0] - x[1]
            w2 = x[0]
            w3 = x[1]
            baryCoords.append([w1, w2, w3])

        return np.array(triIndices), np.array(baryCoords)
    def reconstructLandmarksFromBaryCentric(mesh: trimesh.Trimesh, triIndices: np.ndarray, baryCoords: np.ndarray) -> np.ndarray:
        """
        Reconstruct 3D points from barycentric coordinates and triangle indices.

        Args:
            mesh (trimesh.Trimesh): The source mesh.
            triIndices (np.ndarray): (n,) array of triangle indices.
            baryCoords (np.ndarray): (n, 3) array of barycentric coordinates.

        Returns:
            np.ndarray: (n, 3) array of reconstructed 3D points.
        """
        if triIndices.shape[0] != baryCoords.shape[0]:
            raise ValueError("triIndices and baryCoords must have the same number of entries.")

        # Get the triangles from the mesh
        tris = mesh.triangles[triIndices]  # shape (n, 3, 3)

        # Unpack vertices: V = w1*V0 + w2*V1 + w3*V2
        v0 = tris[:, 0, :]  # (n, 3)
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]

        w1 = baryCoords[:, 0][:, np.newaxis]
        w2 = baryCoords[:, 1][:, np.newaxis]
        w3 = baryCoords[:, 2][:, np.newaxis]

        points = w1 * v0 + w2 * v1 + w3 * v2  # (n, 3)
        return points
    templateFolder = r"H:\Data\Template";
    pelvisReconFolder = r"H:\Data\PelvisBoneRecon";
    pelvisGeoFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    debugFolder = pelvisReconFolder + "/Debugs";

    # Getting template information
    print("Getting template information ...");
    tempShape = sp.readMesh(templateFolder + "/TemplatePelvisCoarseShape.ply");
    tempFeatures = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_picked_points.pp");
    defShape = sp.cloneMesh(tempShape);
    defFeatures = tempFeatures.copy();
    defFeatureBaryIndices, defFeatureBaryCoords = computeBarycentricLandmarks(defShape, defFeatures);

    # Reading target geometries
    print("Reading target geometries ...");
    targetShape = sp.readMesh(pelvisGeoFolder + "/100131-PelvisBoneShape.ply");
    targetFeatures = sp.read3DPointsFromPPFile(pelvisGeoFolder + "/100131-PelvisBoneMesh_picked_points.pp");

    # Deform using the rigid transform 
    print("Deform using the rigid transform ...");
    svdTransform = sp.estimateRigidSVDTransform(defFeatures, targetFeatures);
    defShape = sp.transformMesh(defShape, svdTransform);
    defFeatures = reconstructLandmarksFromBaryCentric(defShape, defFeatureBaryIndices, defFeatureBaryCoords);

    # Deform with affine transform 
    print("Deform with affine transform ...");
    affineTransform = sp.estimateAffineTransformCPD(defFeatures, targetFeatures);
    defShape = sp.transformMesh(defShape, affineTransform);

    # Deform using the non-rigid ICP registration
    print("Deform using the non-rigid ICP registration ...");
    defShapeVertices = trimesh.registration.nricp_amberg(
        source_mesh=defShape,
        target_geometry=targetShape,
        source_landmarks=(defFeatureBaryIndices, defFeatureBaryCoords),
        target_positions=targetFeatures
    )

    # Deform using the cage-based deformation
    print("Deforming using the cage-based deformation ...");
    defShape.vertices = defShapeVertices;
    defShape.vertices = sp.estimateNearestPointsFromPoints(defShape.vertices, targetShape.vertices);

    # Debugging
    print("Debugging ...");
    sp.saveMeshToPLY(debugFolder + "/defShape.ply", defShape);

    # Finished processing
    print("Finished processing.");
def deformTemplatePelvisToTargetPelvis_allData_usingGlobalFeatureMeshDeformation():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("\t Please input the command as the following: [ProgramName] [StartIndex] [EndIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);
    processingDisk = "G:";
    templateFolder = processingDisk + r"\Data\Template";
    pelvisBoneMuscleTempFolder = templateFolder + "/PelvisBonesMuscles";
    pelvisReconFolder = processingDisk +  r"\Data\PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + r"\FemalePelvisGeometries";
    debugFolder = processingDisk + r"\Data\PelvisBoneRecon\Debugs";

    # Reading processing IDs
    print("Reading processing IDs ...");
    processingIDs = sp.readListOfStrings(pelvisReconFolder + "/FemalePelvisIDs.txt");

    # Getting template information
    print("Getting template information ...");
    tempPelvisBoneMuscleMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMuscles.ply");
    tempPelvisMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh.ply");
    tempPelvisMuscleMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisMuscles.ply");
    tempWithoutJointPelvisMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMeshWithOutJoints.ply");    
    tempLeftSacroiliacJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint.ply");
    tempRightSacroiliacJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint.ply");
    tempPubicJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_PubicJoint.ply");
    
    tempPelvisFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_picked_points.pp");

    pelvisFeatureBaryIndicesOnMesh, pelvisFeatureBaryCoordsOnMesh = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempPelvisFeatures);
    withoutJointPelvisVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempWithoutJointPelvisMesh.vertices, tempPelvisMesh.vertices);
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    pelvisMuscleVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisMuscleMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
        
    tempLeftSacroiliacJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint_picked_points.pp");
    tempRightSacroiliacJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint_picked_points.pp");
    tempPubicJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_PubicJoint_picked_points.pp");
    tempPelvisMuscleFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisMuscles_picked_points.pp");

    leftSacroiliacJointFeatureBaryIndices, leftSacroiliacJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempLeftSacroiliacJointFeatures);
    rightSacroiliacJointFeatureBaryIndices, rightSacroiliacJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempRightSacroiliacJointFeatures);
    pubicJointFeatureBaryIndices, pubicJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempPubicJointFeatures);
    pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisMesh, tempPelvisMuscleFeatures);
    
    leftSacroiliacJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(leftSacroiliacJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
    rightSacroiliacJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(rightSacroiliacJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
    pubicJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(pubicJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
        
    leftSacroiliacJointVertexIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/LeftSacroiliacJointVertexIndices.csv");
    rightSacroiliacJointVertexIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/RightSacroiliacJointVertexIndices.csv");
    pubicJointVertexIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/PubicJointVertexIndices.csv");

    # Checking for each subject
    print("Checking for each subject ...");
    for i in range(startIndex, endIndex + 1):
        # Debugging 
        sID = processingIDs[i];
        print("/****************************************** Processing subject: ", i, " with ID: ", sID);

        # Reading target information
        print("Reading target information ...");
        targetPelvisMesh = sp.readMesh(femalePelvisFolder + f"/{sID}-PelvisBoneMesh.ply");
        targetPelvisFeatures = sp.read3DPointsFromPPFile(femalePelvisFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp");

        # Fixing the target information
        print("Fixing the target information ...");
        targetPelvisMesh = sp.fixMesh(targetPelvisMesh);

        # Prepare the deformation buffer
        print("Preparing the deformation buffer ...");
        deformedWithoutJointPelvisMesh = sp.cloneMesh(tempWithoutJointPelvisMesh);
        deformedLeftSacroiliacJointMesh = sp.cloneMesh(tempLeftSacroiliacJointMesh);
        deformedRightSacroiliacJointMesh = sp.cloneMesh(tempRightSacroiliacJointMesh);
        deformedPubicJointMesh = sp.cloneMesh(tempPubicJointMesh);
        deformedPelvisMuscleMesh = sp.cloneMesh(tempPelvisMuscleMesh);
        defPelFeatures = tempPelvisFeatures.copy();
        
        # Deform using the rigid transform 
        print("Deform using the rigid transform ...");
        svdTransform = sp.estimateRigidSVDTransform(defPelFeatures, targetPelvisFeatures);
        deformedWithoutJointPelvisMesh.vertices = sp.transform3DPoints(deformedWithoutJointPelvisMesh.vertices, svdTransform);
        deformedLeftSacroiliacJointMesh.vertices = sp.transform3DPoints(deformedLeftSacroiliacJointMesh.vertices, svdTransform);
        deformedRightSacroiliacJointMesh.vertices = sp.transform3DPoints(deformedRightSacroiliacJointMesh.vertices, svdTransform);
        deformedPubicJointMesh.vertices = sp.transform3DPoints(deformedPubicJointMesh.vertices, svdTransform);
        deformedPelvisMuscleMesh.vertices = sp.transform3DPoints(deformedPelvisMuscleMesh.vertices, svdTransform);
        defPelFeatures = sp.transform3DPoints(defPelFeatures, svdTransform);
        
        # Deform with affine transform 
        print("Deform with affine transform ...");
        affineTransform = sp.estimateAffineTransformCPD(defPelFeatures, targetPelvisFeatures);

        deformedWithoutJointPelvisMesh.vertices = sp.transform3DPoints(deformedWithoutJointPelvisMesh.vertices, affineTransform);
        deformedLeftSacroiliacJointMesh.vertices = sp.transform3DPoints(deformedLeftSacroiliacJointMesh.vertices, affineTransform);
        deformedRightSacroiliacJointMesh.vertices = sp.transform3DPoints(deformedRightSacroiliacJointMesh.vertices, affineTransform);
        deformedPubicJointMesh.vertices = sp.transform3DPoints(deformedPubicJointMesh.vertices, affineTransform);

        deformedPelvisMesh = sp.cloneMesh(tempPelvisMesh);
        deformedPelvisMesh.vertices[withoutJointPelvisVertexIndices] = deformedWithoutJointPelvisMesh.vertices;
        deformedPelvisMesh.vertices[leftSacroiliacJointVertexIndices] = deformedLeftSacroiliacJointMesh.vertices;
        deformedPelvisMesh.vertices[rightSacroiliacJointVertexIndices] = deformedRightSacroiliacJointMesh.vertices;
        deformedPelvisMesh.vertices[pubicJointVertexIndices] = deformedPubicJointMesh.vertices;

        # Deform using non rigid ICP
        print("Deform using non rigid ICP ...");
        personalizedWithoutJointPelvisVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedWithoutJointPelvisMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(pelvisFeatureBaryIndicesOnMesh, pelvisFeatureBaryCoordsOnMesh),
            target_positions=targetPelvisFeatures
        )

        # Try to deform the joints based on the deformed pelvis mesh
        print("Try to deform the joints based on the deformed pelvis mesh ...");
        personalizedPelvisMesh = sp.cloneMesh(tempPelvisMesh);
        personalizedPelvisMesh.vertices[withoutJointPelvisVertexIndices] = personalizedWithoutJointPelvisVertices;

        deformedLeftSacroiliacJointFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, leftSacroiliacJointFeatureBaryIndices, leftSacroiliacJointFeatureBaryCoords);
        deformedRightSacroiliacJointFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, rightSacroiliacJointFeatureBaryIndices, rightSacroiliacJointFeatureBaryCoords);
        deformedPubicJointFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, pubicJointFeatureBaryIndices, pubicJointFeatureBaryCoords);
        
        personalizedLeftSacroiliacJointFeatures = sp.reconstructLandmarksFromBarycentric(personalizedPelvisMesh, leftSacroiliacJointFeatureBaryIndices, leftSacroiliacJointFeatureBaryCoords);
        personalizedRightSacroiliacJointFeatures = sp.reconstructLandmarksFromBarycentric(personalizedPelvisMesh, rightSacroiliacJointFeatureBaryIndices, rightSacroiliacJointFeatureBaryCoords);
        personalizedPubicJointFeatures = sp.reconstructLandmarksFromBarycentric(personalizedPelvisMesh, pubicJointFeatureBaryIndices, pubicJointFeatureBaryCoords);
        
        deformedToPersonalizedLeftSacroiliacJointFeatureDisplacements =  personalizedLeftSacroiliacJointFeatures - deformedLeftSacroiliacJointFeatures;
        rbf = RBFInterpolator(deformedLeftSacroiliacJointFeatures, deformedToPersonalizedLeftSacroiliacJointFeatureDisplacements, kernel='thin_plate_spline');
        personalizedLeftSacroiliacJointMeshVertexDisplacements = rbf(deformedLeftSacroiliacJointMesh.vertices);
        personalizedLeftSacroiliacJointMesh = sp.cloneMesh(deformedLeftSacroiliacJointMesh);
        personalizedLeftSacroiliacJointMesh.vertices = deformedLeftSacroiliacJointMesh.vertices + personalizedLeftSacroiliacJointMeshVertexDisplacements;
        
        deformedToPersonalizedRightSacroiliacJointFeatureDisplacements = personalizedRightSacroiliacJointFeatures - deformedRightSacroiliacJointFeatures;
        rbf = RBFInterpolator(deformedRightSacroiliacJointFeatures, deformedToPersonalizedRightSacroiliacJointFeatureDisplacements, kernel='thin_plate_spline');
        personalizedRightSacroiliacJointMeshVertexDisplacements = rbf(deformedRightSacroiliacJointMesh.vertices);
        personalizedRightSacroiliacJointMesh = sp.cloneMesh(deformedRightSacroiliacJointMesh);
        personalizedRightSacroiliacJointMesh.vertices = deformedRightSacroiliacJointMesh.vertices + personalizedRightSacroiliacJointMeshVertexDisplacements;
        
        deformedToPersonalizedPubicJointFeatureDisplacements = personalizedPubicJointFeatures - deformedPubicJointFeatures;
        rbf = RBFInterpolator(deformedPubicJointFeatures, deformedToPersonalizedPubicJointFeatureDisplacements, kernel='thin_plate_spline');
        personalizedPubicJointMeshVertexDisplacements = rbf(deformedPubicJointMesh.vertices);
        personalizedPubicJointMesh = sp.cloneMesh(deformedPubicJointMesh);
        personalizedPubicJointMesh.vertices = deformedPubicJointMesh.vertices + personalizedPubicJointMeshVertexDisplacements;
        
        personalizedPelvisMesh.vertices[leftSacroiliacJointVertexIndices] = personalizedLeftSacroiliacJointMesh.vertices;
        personalizedPelvisMesh.vertices[rightSacroiliacJointVertexIndices] = personalizedRightSacroiliacJointMesh.vertices;
        personalizedPelvisMesh.vertices[pubicJointVertexIndices] = personalizedPubicJointMesh.vertices;

        # Try to deform pelvis muscle to the personalized pelvis mesh
        print("Try to deform pelvis muscle to the personalized pelvis mesh ..");
        deformedPelvisMuscleFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords);
        personalizedPelvisMusceFeatures = sp.reconstructLandmarksFromBarycentric(personalizedPelvisMesh, pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords);
        deformedToPersonalizedPelvisMuscleFeatureDisplacements = personalizedPelvisMusceFeatures - deformedPelvisMuscleFeatures;
        rbf = RBFInterpolator(deformedPelvisMuscleFeatures, deformedToPersonalizedPelvisMuscleFeatureDisplacements, kernel='thin_plate_spline');        
        personalizedPelvisMuscleVertexDisplacements = rbf(deformedPelvisMuscleMesh.vertices);
        personalizedPelvisMuscleMesh = sp.cloneMesh(deformedPelvisMuscleMesh);
        personalizedPelvisMuscleMesh.vertices = deformedPelvisMuscleMesh.vertices + personalizedPelvisMuscleVertexDisplacements;

        # Comptue the personalized pelvis muscle meshes
        print("Computing the personalized pelvis muscle meshes ...");
        personalizedPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
        personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices] = personalizedPelvisMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[pelvisMuscleVertexIndices] = personalizedPelvisMuscleMesh.vertices;
       
        # Save the personalized mesh
        print("Save the personalized mesh ...");
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh_GB.ply", personalizedPelvisBoneMuscleMesh)
        
    # Finished 
    print("Finished.");
def deformTemplatePelvisToTargetPelvis_allData_usingROIFeatureShapeAndMeshDeformation():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("\t Please input the command as the following: [ProgramName] [StartIndex] [EndIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);    
    processingDisk = "H:/SpinalPelvisPred";
    templateFolder = processingDisk + "/Data/Template";
    pelvisBoneMuscleTempFolder = templateFolder + "/PelvisBonesMuscles";
    pelvisReconFolder = processingDisk +  "/Data/PelvisBoneRecon";
    targetPelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries/1KPelvisData";
    debugFolder = processingDisk + "/Data/PelvisBoneRecon/Debugs";

    # Reading processing IDs
    print("Reading processing IDs ...");
    processingIDs = sp.readListOfStrings(pelvisReconFolder + "/FemalePelvisGeometries/1KPelvisDataFemalePelvisIDs.txt");

    # Getting template information
    print("Getting template information ...");
    tempPelvisBoneMuscleMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMuscles.ply");
    tempPelvisMuscleMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisMuscles.ply");
    tempPelvisShape = sp.readMesh(pelvisBoneMuscleTempFolder + "/TemplatePelvisCoarseShape.ply");
    tempPelvisMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh.ply");
    tempWithoutJointPelvisMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMeshWithOutJoints.ply");
    tempLeftIliumMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftIlium.ply");
    tempRightIliumMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightIlium.ply");
    tempSacrumMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_Sacrum.ply");
    tempLeftSacroiliacJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint.ply");
    tempRightSacroiliacJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint.ply");
    tempPubicJointMesh = sp.readMesh(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_PubicJoint.ply");
    
    tempPelvisFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_picked_points.pp");
    pelvisFeatureBaryIndicesOnShape, pelvisFeatureBaryCoordsOnShape = sp.computeBarycentricLandmarks(tempPelvisShape, tempPelvisFeatures);
    pelvisFeatureBaryIndicesOnMesh, pelvisFeatureBaryCoordsOnMesh = sp.computeBarycentricLandmarks(tempPelvisMesh, tempPelvisFeatures);
    
    leftIliumFeatureIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/LeftIliumFeatureIndices.csv");
    rightIliumFeatureIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/RightIliumFeatureIndices.csv");
    sacrumFeatureIndices = sp.readIndicesFromCSVFile(pelvisBoneMuscleTempFolder + "/SacrumFeatureIndices.csv");
    
    tempLeftSacroiliacJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint_picked_points.pp");
    tempRightSacroiliacJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint_picked_points.pp");
    tempPubicJointFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisBoneMesh_PubicJoint_picked_points.pp");
    tempPelvisMuscleFeatures = sp.read3DPointsFromPPFile(pelvisBoneMuscleTempFolder + "/TempPelvisMuscles_picked_points.pp");

    leftSacroiliacJointFeatureBaryIndices, leftSacroiliacJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempLeftSacroiliacJointFeatures);
    rightSacroiliacJointFeatureBaryIndices, rightSacroiliacJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempRightSacroiliacJointFeatures);
    pubicJointFeatureBaryIndices, pubicJointFeatureBaryCoords = sp.computeBarycentricLandmarks(tempWithoutJointPelvisMesh, tempPubicJointFeatures);
    pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisMesh, tempPelvisMuscleFeatures);
    
    leftSacroiliacJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(leftSacroiliacJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
    rightSacroiliacJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(rightSacroiliacJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
    pubicJointFeatureBaryIndices = sp.transferFaceIndicesToOtherMesh(pubicJointFeatureBaryIndices, tempWithoutJointPelvisMesh, tempPelvisMesh);
        
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    pelvisMuscleVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisMuscleMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    leftIliumVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempLeftIliumMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    rightIliumVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempRightIliumMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    sacrumVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempSacrumMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    leftSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempLeftSacroiliacJointMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    rightSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempRightSacroiliacJointMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    pubicJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPubicJointMesh.vertices, tempPelvisBoneMuscleMesh.vertices);    
    
    # Checking for each subject
    print("Checking for each subject ...");
    for i in range(startIndex, endIndex + 1):
        # Debugging 
        sID = processingIDs[i];
        print("/****************************************** Processing subject: ", i, " with ID: ", sID);

        # Reading target information
        print("Reading target information ...");
        targetPelvisShape = sp.readMesh(targetPelvisFolder + f"/{sID}-PelvisBoneShape.ply");
        targetPelvisMesh = sp.readMesh(targetPelvisFolder + f"/{sID}-PelvisBoneMesh.ply");
        targetPelvisFeatures = sp.read3DPointsFromPPFile(targetPelvisFolder + f"/{sID}-PelvisBoneMesh_picked_points.pp");

        # Fixing the target information
        print("Fixing the target information ...");
        targetPelvisShape = sp.fixMesh(targetPelvisShape);
        targetPelvisMesh = sp.fixMesh(targetPelvisMesh);

        # Prepare the aligned buffers
        print("Preparing the aligned buffers ...");
        alignedPelvisShape = sp.cloneMesh(tempPelvisShape);
        alignedPelvisFeatures = tempPelvisFeatures.copy();
        
        # Deform using the rigid transform 
        print("Deform using the rigid transform ...");
        svdTransform = sp.estimateRigidSVDTransform(alignedPelvisFeatures, targetPelvisFeatures);
        alignedPelvisShape.vertices = sp.transform3DPoints(alignedPelvisShape.vertices, svdTransform);
        alignedPelvisFeatures = sp.reconstructLandmarksFromBarycentric(alignedPelvisShape, pelvisFeatureBaryIndicesOnShape, pelvisFeatureBaryCoordsOnShape);
        
        # Deform with affine transform 
        print("Deform with affine transform ...");
        affineTransform = sp.estimateAffineTransformCPD(alignedPelvisFeatures, targetPelvisFeatures);
        alignedPelvisShape = sp.transformMesh(alignedPelvisShape, affineTransform);
        del(alignedPelvisFeatures); gc.collect();
        
        # Deform using the non-rigid ICP registration
        print("Deform using the non-rigid ICP registration ...");
        alignedPelvisShapeVertices = trimesh.registration.nricp_amberg(
            source_mesh=alignedPelvisShape,
            target_geometry=targetPelvisShape,
            source_landmarks=(pelvisFeatureBaryIndicesOnShape, pelvisFeatureBaryCoordsOnShape),
            target_positions=targetPelvisFeatures
        )
        alignedPelvisShape.vertices = alignedPelvisShapeVertices.copy(); 
        del(alignedPelvisShapeVertices); gc.collect();

        # Project deformed shape to the target shape
        print("Project deformed shape to the taret shape ...");
        alignedPelvisShape = sp.projectMeshOntoMesh(alignedPelvisShape, targetPelvisShape);        

        # Deform the template pelvis mesh to def pelvis shape
        print("Deform template pelvis mesh to def pelvis shape ...");
        print("\t Using the rigid transform ...");
        deformedPelvisShape = sp.cloneMesh(tempPelvisShape);
        deformedPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
        svdTransform = sp.estimateRigidSVDTransform(deformedPelvisShape.vertices, alignedPelvisShape.vertices);
        deformedPelvisShape.vertices = sp.transform3DPoints(deformedPelvisShape.vertices, svdTransform);
        deformedPelvisBoneMuscleMesh.vertices = sp.transform3DPoints(deformedPelvisBoneMuscleMesh.vertices, svdTransform);

        print("\t Using the non-rigid transform ...");
        affineTransform = sp.estimateAffineTransformCPD(deformedPelvisShape.vertices, alignedPelvisShape.vertices);
        deformedPelvisShape.vertices = sp.transform3DPoints(deformedPelvisShape.vertices, affineTransform);
        deformedPelvisBoneMuscleMesh.vertices = sp.transform3DPoints(deformedPelvisBoneMuscleMesh.vertices, affineTransform);

        print("\t Using the radial basic function ...");
        deformedToAlignedShapeVertexDisplacements = alignedPelvisShape.vertices - deformedPelvisShape.vertices;
        rbf = RBFInterpolator(deformedPelvisShape.vertices, deformedToAlignedShapeVertexDisplacements, kernel='thin_plate_spline', smoothing=1e-3);
        deformedPelvisBoneMuscleVertexDisplacements = rbf(deformedPelvisBoneMuscleMesh.vertices);
        deformedPelvisBoneMuscleMesh.vertices = deformedPelvisBoneMuscleMesh.vertices + deformedPelvisBoneMuscleVertexDisplacements;
        del(deformedPelvisShape, deformedToAlignedShapeVertexDisplacements, svdTransform, affineTransform, rbf, deformedPelvisBoneMuscleVertexDisplacements); gc.collect();
        
        # Try to deform the pelvis mesh using the ROI shape
        print("Try to deform the pelvis mesh using the ROI shape ...");
        ## Getting the ROI meshes
        print("\t Getting the ROI meshes ...");
        deformedPelvisMesh = sp.cloneMesh(tempPelvisMesh);
        deformedPelvisMuscleMesh = sp.cloneMesh(tempPelvisMuscleMesh);
        deformedLeftIliumMesh = sp.cloneMesh(tempLeftIliumMesh);
        deformedRightIliumMesh = sp.cloneMesh(tempRightIliumMesh);
        deformedSacrumMesh = sp.cloneMesh(tempSacrumMesh);
        deformedLeftSacroiliacJointMesh = sp.cloneMesh(tempLeftSacroiliacJointMesh);
        deformedRightSacroiliacJointMesh = sp.cloneMesh(tempRightSacroiliacJointMesh);
        deformedPubicJointMesh = sp.cloneMesh(tempPubicJointMesh);        
        
        deformedPelvisMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
        deformedPelvisMuscleMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pelvisMuscleVertexIndices];
        deformedLeftIliumMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[leftIliumVertexIndices];
        deformedRightIliumMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[rightIliumVertexIndices];
        deformedSacrumMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[sacrumVertexIndices];
        deformedLeftSacroiliacJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[leftSacroiliacJointVertexIndices];
        deformedRightSacroiliacJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[rightSacroiliacJointVertexIndices];
        deformedPubicJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pubicJointVertexIndices];

        ## Getting ROI features
        print("\t Getting the ROI features ...");
        deformedFullPelvisFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, pelvisFeatureBaryIndicesOnMesh, pelvisFeatureBaryCoordsOnMesh);
        deformedLeftIliumFeatures = deformedFullPelvisFeatures[leftIliumFeatureIndices];
        deformedRightIliumFeatures = deformedFullPelvisFeatures[rightIliumFeatureIndices];
        deformedSacrumFeatures = deformedFullPelvisFeatures[sacrumFeatureIndices];        

        ## Compute bary coords for ROI features
        print("\t Compute bary coords for ROI features ...");
        leftIliumFeatureBaryIndices, leftIliumFeatureBaryCoords = sp.computeBarycentricLandmarks(deformedLeftIliumMesh, deformedLeftIliumFeatures);
        rightIliumFeatureBaryIndices, rightIliumFeatureBaryCoords = sp.computeBarycentricLandmarks(deformedRightIliumMesh, deformedRightIliumFeatures);
        sacrumFeatureBaryIndices, sacrumFeatureBaryCoords = sp.computeBarycentricLandmarks(deformedSacrumMesh, deformedSacrumFeatures);
        
        ## Compute the target ROI features
        print("\t Compute target ROI features ...");
        targetLeftIliumFeatures = targetPelvisFeatures[leftIliumFeatureIndices];
        targetRightIliumFeatures = targetPelvisFeatures[rightIliumFeatureIndices];
        targetSacrumFeatures = targetPelvisFeatures[sacrumFeatureIndices];

        ## Deform leftIliumMesh to the target mesh
        print("\t Deform leftIliumMesh to the target mesh ...");
        personalizedLeftIliumMeshVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedLeftIliumMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(leftIliumFeatureBaryIndices, leftIliumFeatureBaryCoords),
            target_positions=targetLeftIliumFeatures
        )
        personalizedLeftIliumMesh = sp.cloneMesh(deformedLeftIliumMesh);
        personalizedLeftIliumMesh.vertices = personalizedLeftIliumMeshVertices.copy();
        del(personalizedLeftIliumMeshVertices); gc.collect();

        ## Deform rightIliumMesh to the target mesh
        print("\t Deform rightIliumMesh to the target mesh ...");
        personalizedRightIliumMeshVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedRightIliumMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(rightIliumFeatureBaryIndices, rightIliumFeatureBaryCoords),
            target_positions=targetRightIliumFeatures
        )
        personalizedRightIliumMesh = sp.cloneMesh(deformedRightIliumMesh);
        personalizedRightIliumMesh.vertices = personalizedRightIliumMeshVertices.copy();
        del(personalizedRightIliumMeshVertices); gc.collect();

        ## Deform sacrum to the target mesh
        print("\t Deform sacrum to the target mesh ...");
        personalizedSacrumMeshVertices = trimesh.registration.nricp_amberg(
            source_mesh=deformedSacrumMesh,
            target_geometry=targetPelvisMesh,
            source_landmarks=(sacrumFeatureBaryIndices, sacrumFeatureBaryCoords),
            target_positions=targetSacrumFeatures
        )
        personalizedSacrumMesh = sp.cloneMesh(deformedSacrumMesh);
        personalizedSacrumMesh.vertices = personalizedSacrumMeshVertices.copy();
        del(personalizedSacrumMeshVertices); gc.collect();

        ## Forming personalized pelvis bone muscle mesh
        print("\t Forming personalized pelvis bone muscle mesh ...");
        personalizedPelvisBoneMuscleMesh = sp.cloneMesh(deformedPelvisBoneMuscleMesh);
        personalizedPelvisBoneMuscleMesh.vertices[leftIliumVertexIndices] = personalizedLeftIliumMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[rightIliumVertexIndices] = personalizedRightIliumMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[sacrumVertexIndices] = personalizedSacrumMesh.vertices;
        personalizedPelvisMesh = sp.cloneMesh(deformedPelvisMesh);
        personalizedPelvisMesh.vertices = personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

        # Deform the pelvis muscle
        print("Deforming the pelvis muscle ...");
        deformedPelvisMuscleFeatures = sp.reconstructLandmarksFromBarycentric(deformedPelvisMesh, pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords);
        personalizedPelvisMusceFeatures = sp.reconstructLandmarksFromBarycentric(personalizedPelvisMesh, pelvisMuscleFeatureBaryIndices, pelvisMuscleFeatureBaryCoords);
        deformedToPersonalizedPelvisMuscleFeatureDisplacements = personalizedPelvisMusceFeatures - deformedPelvisMuscleFeatures;
        rbf = RBFInterpolator(deformedPelvisMuscleFeatures, 
                              deformedToPersonalizedPelvisMuscleFeatureDisplacements, 
                              kernel='thin_plate_spline', 
                              smoothing=1e-3);
        personalizedPelvisMuscleVertexDisplacements = rbf(deformedPelvisMuscleMesh.vertices);
        personalizedPelvisMuscleMesh = sp.cloneMesh(deformedPelvisMuscleMesh);
        personalizedPelvisMuscleMesh.vertices = deformedPelvisMuscleMesh.vertices + personalizedPelvisMuscleVertexDisplacements;
        del(deformedPelvisMuscleFeatures, personalizedPelvisMusceFeatures, deformedToPersonalizedPelvisMuscleFeatureDisplacements, 
            rbf, personalizedPelvisMuscleVertexDisplacements); gc.collect();

        # Comptue the personalized pelvis muscle meshes
        print("Computing the personalized pelvis muscle meshes ...");
        personalizedPelvisBoneMuscleMesh = sp.cloneMesh(deformedPelvisBoneMuscleMesh);
        personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices] = personalizedPelvisMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[pelvisMuscleVertexIndices] = personalizedPelvisMuscleMesh.vertices;
        
        # Save the personalized results
        print("Save the personalized results ...");
        sp.saveMeshToPLY(debugFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply", personalizedPelvisBoneMuscleMesh);
        del(personalizedPelvisBoneMuscleMesh); gc.collect();

    # Finished 
    print("Finished.");
def refineThePelvisJointsForAllData():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("\t Please input the command as the following: [ProgramName] [StartSubjectIndex] [EndSubjectIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);
    disk = "E:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    femalePelvisFolder = mainFolder + r"\FemalePelvisGeometries";
    personalizedFolder = mainFolder + r"\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templatePelvisFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    debugFolder = mainFolder + r"\Debugs";

    # Reading subject IDs
    print("Reading subject IDs ...");
    subjectIDFilePath = femalePelvisFolder + r"\FemalePelvisIDs.txt";
    if not os.path.exists(subjectIDFilePath):
        print("\t Subject ID file does not exist: ", subjectIDFilePath); return;
    subjectIDs = sp.readListOfStrings(subjectIDFilePath);
    print("\t Number of subject IDs: ", len(subjectIDs));

    # Reading template data
    print("Reading template data ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templatePelvisFolder + "/TempPelvisBoneMuscles.ply");
    templateLeftSacroiliacJointMesh = sp.readMesh(templatePelvisFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint.ply");
    templateRightSacroiliacJointMesh = sp.readMesh(templatePelvisFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint.ply");
    templatePubicJointMesh = sp.readMesh(templatePelvisFolder + "/TempPelvisBoneMesh_PubicJoint.ply");
    templateWithoutJointPelvisBoneMesh = sp.readMesh(templatePelvisFolder + "/TempWithoutJointPelvisBoneMesh.ply");
    
    # Compute ROI indices
    print("Compute ROI indices ...");
    leftSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(templateLeftSacroiliacJointMesh.vertices, templatePelvisBoneMuscleMesh.vertices);
    rightSacroiliacJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(templateRightSacroiliacJointMesh.vertices, templatePelvisBoneMuscleMesh.vertices);
    pubicJointVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePubicJointMesh.vertices, templatePelvisBoneMuscleMesh.vertices);
    withoutJointPelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templateWithoutJointPelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(startIndex, endIndex + 1):       
        # Debugging
        sID = subjectIDs[i];
        print("/****************************************** Processing subject: ", sID);

        # Reading personalized pelvis bone muscle mesh
        print("\t Reading personalized pelvis bone muscle mesh ...");
        personalizedPelvisBoneMuscleMeshFilePath = personalizedFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply";
        if not os.path.exists(personalizedPelvisBoneMuscleMeshFilePath):
            print("\t Personalized pelvis bone muscle mesh file does not exist: ", personalizedPelvisBoneMuscleMeshFilePath);
            return;
        personalizedPelvisBoneMuscleMesh = sp.readMesh(personalizedPelvisBoneMuscleMeshFilePath);
        personalizedWithoutJointPelvisBoneMesh = sp.cloneMesh(templateWithoutJointPelvisBoneMesh);
        personalizedWithoutJointPelvisBoneMesh.vertices = personalizedPelvisBoneMuscleMesh.vertices[withoutJointPelvisBoneVertexIndices];
    
        # Deform using svd transform
        print("\t Deform using svd transform ...");
        deformedPelvisBoneMuscleMesh = sp.cloneMesh(templatePelvisBoneMuscleMesh);
        deformedWithoutJointPelvisBoneMesh = sp.cloneMesh(templateWithoutJointPelvisBoneMesh);
        deformedLeftScaroiliacJointMesh = sp.cloneMesh(templateLeftSacroiliacJointMesh);
        deformedRightScaroiliacJointMesh = sp.cloneMesh(templateRightSacroiliacJointMesh);
        deformedPubicJointMesh = sp.cloneMesh(templatePubicJointMesh);
        svdTransform = sp.estimateRigidSVDTransform(deformedWithoutJointPelvisBoneMesh.vertices, personalizedWithoutJointPelvisBoneMesh.vertices);
        deformedPelvisBoneMuscleMesh = sp.transformMesh(deformedPelvisBoneMuscleMesh, svdTransform);
        deformedWithoutJointPelvisBoneMesh = sp.transformMesh(deformedWithoutJointPelvisBoneMesh, svdTransform);
        deformedLeftScaroiliacJointMesh = sp.transformMesh(deformedLeftScaroiliacJointMesh, svdTransform);
        deformedRightScaroiliacJointMesh = sp.transformMesh(deformedRightScaroiliacJointMesh, svdTransform);
        deformedPubicJointMesh = sp.transformMesh(deformedPubicJointMesh, svdTransform);

        # Deform using affine trasform
        print("\t Deform using affine transform ...");
        affineTransform = sp.estimateAffineTransformCPD(deformedWithoutJointPelvisBoneMesh.vertices, personalizedWithoutJointPelvisBoneMesh.vertices);
        deformedPelvisBoneMuscleMesh = sp.transformMesh(deformedPelvisBoneMuscleMesh, affineTransform);
        deformedWithoutJointPelvisBoneMesh = sp.transformMesh(deformedWithoutJointPelvisBoneMesh, affineTransform);
        deformedLeftScaroiliacJointMesh = sp.transformMesh(deformedLeftScaroiliacJointMesh, affineTransform);
        deformedRightScaroiliacJointMesh = sp.transformMesh(deformedRightScaroiliacJointMesh, affineTransform);
        deformedPubicJointMesh = sp.transformMesh(deformedPubicJointMesh, affineTransform);

        # Deform using the radial basic function
        print("\t Deform using the radial basic function ...");
        deformedToPersonalizedWithoutJointPelvisBoneMeshVertexDisplacements = personalizedWithoutJointPelvisBoneMesh.vertices - deformedWithoutJointPelvisBoneMesh.vertices;
        rbf = RBFInterpolator(deformedWithoutJointPelvisBoneMesh.vertices, 
                              deformedToPersonalizedWithoutJointPelvisBoneMeshVertexDisplacements,
                              kernel='linear', epsilon=10.0, smoothing=5.0, degree=1, neighbors=150);
        deformedPelvisMuscleVertexDisplacements = rbf(deformedPelvisBoneMuscleMesh.vertices);
        deformedPelvisBoneMuscleMesh.vertices = deformedPelvisBoneMuscleMesh.vertices + deformedPelvisMuscleVertexDisplacements;
        deformedWithoutJointPelvisBoneMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[withoutJointPelvisBoneVertexIndices];
        deformedLeftScaroiliacJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[leftSacroiliacJointVertexIndices];
        deformedRightScaroiliacJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[rightSacroiliacJointVertexIndices];
        deformedPubicJointMesh.vertices = deformedPelvisBoneMuscleMesh.vertices[pubicJointVertexIndices];

        # Forming the personalized pelvis bone muscle mesh
        print("\t Forming the personalized pelvis bone muscle mesh ...");
        personalizedPelvisBoneMuscleMesh.vertices[leftSacroiliacJointVertexIndices] = deformedLeftScaroiliacJointMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[rightSacroiliacJointVertexIndices] = deformedRightScaroiliacJointMesh.vertices;
        personalizedPelvisBoneMuscleMesh.vertices[pubicJointVertexIndices] = deformedPubicJointMesh.vertices;
        
        # Save the personalized pelvis bone muscle mesh
        print("\t Save the personalized pelvis bone muscle mesh ...");
        personalizedPelvisBoneMuscleMeshFilePath = debugFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply";
        sp.saveMeshToPLY(personalizedPelvisBoneMuscleMeshFilePath, personalizedPelvisBoneMuscleMesh);

    # Finished processing
    print("Finished processing.");
def normalizePersonalizedPelvisBoneMuscleMesh():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("\t Please input the command as the following: [ProgramName] [StartSubjectIndex] [EndSubjectIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    femalePelvisFolder = mainFolder + r"\FemalePelvisGeometries";
    personalizedFolder = mainFolder + r"\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templatePelvisFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    debugFolder = mainFolder + r"\Debugs";

    # Reading subject IDs
    print("Reading subject IDs ...");
    subjectIDFilePath = femalePelvisFolder + r"\FemalePelvisIDs.txt";
    if not os.path.exists(subjectIDFilePath):
        print("\t Subject ID file does not exist: ", subjectIDFilePath); return;
    subjectIDs = sp.readListOfStrings(subjectIDFilePath);
    print("\t Number of subject IDs: ", len(subjectIDs));

    # Normalize first personalized pelvis bone muscle mesh to the origin of the coordinate system
    print("Normalize first personalized pelvis bone muscle mesh to the origin of the coordinate system ...");
    firstPersonalizedPelvisBoneMuscleMeshFilePath = personalizedFolder + f"/{subjectIDs[0]}-PersonalizedPelvisBoneMuscleMesh.ply";
    if not os.path.exists(firstPersonalizedPelvisBoneMuscleMeshFilePath):
        print("\t First personalized pelvis bone muscle mesh file does not exist: ", firstPersonalizedPelvisBoneMuscleMeshFilePath);
        return;
    firstPersonalizedPelvisBoneMuscleMesh = sp.readMesh(firstPersonalizedPelvisBoneMuscleMeshFilePath);
    firstPersonalizedPelvisBoneMuscleCentroid = sp.computeCentroidPoint(firstPersonalizedPelvisBoneMuscleMesh.vertices);
    originCoordinateSystem = np.array([0, 0, 0]);
    svdTransform = sp.estimateTranslationTransformFromPointToPoint(firstPersonalizedPelvisBoneMuscleCentroid, originCoordinateSystem);
    firstPersonalizedPelvisBoneMuscleMesh = sp.transformMesh(firstPersonalizedPelvisBoneMuscleMesh, svdTransform);

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(startIndex, endIndex + 1):
        # Debugging
        sID = subjectIDs[i];
        print("/****************************************** Processing subject: ", sID);
    
        # Estimate rigid transform from mesh i to the first mesh
        personalizedPelvisBoneMuscleMeshFilePath = personalizedFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply";
        if not os.path.exists(personalizedPelvisBoneMuscleMeshFilePath):
            print("\t Personalized pelvis bone muscle mesh file does not exist: ", personalizedPelvisBoneMuscleMeshFilePath);
            return;
        personalizedPelvisBoneMuscleMesh = sp.readMesh(personalizedPelvisBoneMuscleMeshFilePath);
        svdTransform = sp.estimateRigidSVDTransform(personalizedPelvisBoneMuscleMesh.vertices, firstPersonalizedPelvisBoneMuscleMesh.vertices);
        personalizedPelvisBoneMuscleMesh = sp.transformMesh(personalizedPelvisBoneMuscleMesh, svdTransform);

        # Save transformed mesh to out folder
        personalizedPelvisBoneMuscleMeshFilePath = debugFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply";
        sp.saveMeshToPLY(personalizedPelvisBoneMuscleMeshFilePath, personalizedPelvisBoneMuscleMesh);

    # Finished processing
    print("Finished processing.");
def normalizeAllPersonalizedPelvisBoneMuscleMesh():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("\t Please input the command as the following: [ProgramName] [StartSubjectIndex] [EndSubjectIndex]"); return;
    startIndex = int(sys.argv[1]); endIndex = int(sys.argv[2]);
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    femalePelvisFolder = mainFolder + r"\FemalePelvisGeometries";
    personalizedFolder = mainFolder + r"\FemalePelvisGeometries\PersonalizedPelvisStructures";
    templatePelvisFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    debugFolder = mainFolder + r"\Debugs";

    # Reading subject IDs
    print("Reading subject IDs ...");
    subjectIDFilePath = femalePelvisFolder + r"\FemalePelvisIDs.txt";
    if not os.path.exists(subjectIDFilePath):
        print("\t Subject ID file does not exist: ", subjectIDFilePath); return;
    subjectIDs = sp.readListOfStrings(subjectIDFilePath);
    print("\t Number of subject IDs: ", len(subjectIDs));

    # Normalize first personalized pelvis bone muscle mesh to the origin of the coordinate system
    print("Normalize first personalized pelvis bone muscle mesh to the origin of the coordinate system ...");
    firstPersonalizedPelvisBoneMuscleMeshFilePath = personalizedFolder + f"/{subjectIDs[0]}-PersonalizedPelvisBoneMuscleMesh.ply";
    if not os.path.exists(firstPersonalizedPelvisBoneMuscleMeshFilePath):
        print("\t First personalized pelvis bone muscle mesh file does not exist: ", firstPersonalizedPelvisBoneMuscleMeshFilePath);
        return;
    firstPersonalizedPelvisBoneMuscleMesh = sp.readMesh(firstPersonalizedPelvisBoneMuscleMeshFilePath);
    firstPersonalizedPelvisBoneMuscleCentroid = sp.computeCentroidPoint(firstPersonalizedPelvisBoneMuscleMesh.vertices);
    originCoordinateSystem = np.array([0, 0, 0]);
    svdTransform = sp.estimateTranslationTransformFromPointToPoint(firstPersonalizedPelvisBoneMuscleCentroid, originCoordinateSystem);
    firstPersonalizedPelvisBoneMuscleMesh = sp.transformMesh(firstPersonalizedPelvisBoneMuscleMesh, svdTransform);

    # Processing for each subject
    print("Processing for each subject ...");
    for i in range(startIndex, endIndex + 1):
        # Debugging
        sID = subjectIDs[i];
        print("/****************************************** Processing subject: ", sID);
    
        # Estimate rigid transform from mesh i to the first mesh
        personalizedPelvisBoneMuscleMeshFilePath = personalizedFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply";
        if not os.path.exists(personalizedPelvisBoneMuscleMeshFilePath):
            print("\t Personalized pelvis bone muscle mesh file does not exist: ", personalizedPelvisBoneMuscleMeshFilePath);
            return;
        personalizedPelvisBoneMuscleMesh = sp.readMesh(personalizedPelvisBoneMuscleMeshFilePath);
        svdTransform = sp.estimateRigidSVDTransform(personalizedPelvisBoneMuscleMesh.vertices, firstPersonalizedPelvisBoneMuscleMesh.vertices);
        personalizedPelvisBoneMuscleMesh = sp.transformMesh(personalizedPelvisBoneMuscleMesh, svdTransform);

        # Save transformed mesh to out folder
        personalizedPelvisBoneMuscleMeshFilePath = debugFolder + f"/{sID}-PersonalizedPelvisBoneMuscleMesh.ply";
        sp.saveMeshToPLY(personalizedPelvisBoneMuscleMeshFilePath, personalizedPelvisBoneMuscleMesh);

    # Finished processing
    print("Finished processing.");
def prepareTrainingTestingIDs():
    # Initializing
    print("Initializing ...");
    def splitData(data, trainRatio=0.7, valRatio=0.2, testRatio=0.1, seed=42):
        random.seed(seed)
        random.shuffle(data)

        trainEnd = int(trainRatio * len(data))
        valEnd = trainEnd + int(valRatio * len(data))

        trainSet = data[:trainEnd]
        valSet = data[trainEnd:valEnd]
        testSet = data[valEnd:]

        return trainSet, valSet, testSet
    disk = "I:";
    subjectIDFilePath = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\FemalePelvisGeometries\FemalePelvisIDs.txt";
    debugFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\Debugs";

    # Reading subject IDs
    print("Reading subject IDs ...");
    subjectIDs = sp.readListOfStrings(subjectIDFilePath);

    # Train test split
    print("Train test splits ...");
    numOfValids = 10;
    for i in range(numOfValids):
        # Debugging
        print("\t Processing the valid: ", i);

        # Generate the training testing validating
        trainingIDs, validIDs, testingIDs = splitData(subjectIDs, 0.7, 0.2, 0.1, seed=i);

        # Save the training testing and validating
        sp.saveListOfStrings(debugFolder + f"/TrainingIDs_{i}.txt", trainingIDs);
        sp.saveListOfStrings(debugFolder + f"/ValidationIDs_{i}.txt", validIDs);
        sp.saveListOfStrings(debugFolder + f"/TestingIDs_{i}.txt", testingIDs);
    
    # Finished processing
    print("Finished processing.");
def manageDataForSystemDatabase():
    # Initializing
    print("Initializing ...");
    disk = "G:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/" 
    templateFolder = disk + r"\Data\Template\PelvisBonesMuscles";
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";
    pelvisBoneMuscleTemplateDataFieldName = "PelvisBoneMuscleTemplateData";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Adding template data to the system database
    print("Adding template data to the system database ...");
    def addingTemplatePelvicBoneMuscleDataToTheSystemDatabase():
        print("Preparing data for writing ...");
        leftIliumFeatureIndices = sp.readIndicesFromCSVFile(templateFolder + "/LeftIliumFeatureIndices.csv");
        leftIliumVertexIndices = sp.readIndicesFromCSVFile(templateFolder + "/LeftIliumVertexIndices.csv");
        leftSacroiliacJointVertexIndices = sp.readIndicesFromCSVFile(templateFolder + "/LeftSacroiliacJointVertexIndices.csv");
        pubicJointVertexIndices = sp.readIndicesFromCSVFile(templateFolder + "/PubicJointVertexIndices.csv");
        rightIliumFeatureIndices = sp.readIndicesFromCSVFile(templateFolder + "/RightIliumFeatureIndices.csv");
        rightIliumVertexIndices = sp.readIndicesFromCSVFile(templateFolder + "/RightIliumVertexIndices.csv");
        rightSacroiliacJointVertexIndices = sp.readIndicesFromCSVFile(templateFolder + "/RightSacroiliacJointVertexIndices.csv");
        sacrumFeatureIndices = sp.readIndicesFromCSVFile(templateFolder + "/SacrumFeatureIndices.csv");
        sacrumVertexIndices = sp.readIndicesFromCSVFile(templateFolder + "/SacrumVertexIndices.csv");
        templatePelvisCoarseShape = sp.readMesh(templateFolder + "/TemplatePelvisCoarseShape.ply");
        tempPelvisBoneMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");
        tempPelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMuscles.ply");
        tempPelvisMuscleMesh = sp.readMesh(templateFolder + "/TempPelvisMuscles.ply");
        tempPelvisBoneMeshIliumJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_IliumJoint.ply");
        tempPelvisBoneMeshLeftIlium = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_LeftIlium.ply");
        tempPelvisBoneMeshLeftSacroiliacJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint.ply");
        tempPelvisBoneMeshLeftSacroiliacJointPickedPoints = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_LeftSacroiliacJoint_picked_points.pp");
        tempPelvisBoneMeshPickedPoints = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_picked_points.pp");
        tempPelvisBoneMeshLeftIliumPickedPoints = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points_leftIlium.off");
        tempPelvisBoneMeshRightIliumPickedPoints = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points_rightIlium.off");
        tempPelvisBoneMeshSacrumPickedPoints = sp.read3DPointsFromOFFFile(templateFolder + "/TempPelvisBoneMesh_picked_points_sacrum.off");
        tempPelvisBoneMeshPubicJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_PubicJoint.ply");
        tempPelvisBoneMeshPubicJointPickedPoints = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_PubicJoint_picked_points.pp");
        tempPelvisBoneMeshRightIlium = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_RightIlium.ply");
        tempPelvisBoneMeshRightSacroiliacJoint = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint.ply");
        tempPelvisBoneMeshRightSacroiliacJointPickedPoints = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisBoneMesh_RightSacroiliacJoint_picked_points.pp");
        tempPelvisBoneMeshSacrum = sp.readMesh(templateFolder + "/TempPelvisBoneMesh_Sacrum.ply");
        tempPelvisBoneMeshWithOutJoints = sp.readMesh(templateFolder + "/TempPelvisBoneMeshWithOutJoints.ply");
        tempPelvisBoneShape = sp.readMesh(templateFolder + "/TempPelvisBoneShape.ply");
        tempPelvisMusclesPickedPoints = sp.read3DPointsFromPPFile(templateFolder + "/TempPelvisMuscles_picked_points.pp");
        
        # Remove the pelvis bone muscle template data field
        print("Removing the template field ...");
        database.removeField(pelvisBoneMuscleTemplateDataFieldName);

        # Adding new data
        print("Adding new data ...");
        database.addField("PelvisBoneMuscleTemplateData");
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "LeftIliumFeatureIndices", leftIliumFeatureIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "LeftIliumVertexIndices", leftIliumVertexIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "LeftSacroiliacJointVertexIndices", leftSacroiliacJointVertexIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "PubicJointVertexIndices", pubicJointVertexIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "RightIliumFeatureIndices", rightIliumFeatureIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "RightIliumVertexIndices", rightIliumVertexIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "RightSacroiliacJointVertexIndices", rightSacroiliacJointVertexIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "SacrumFeatureIndices", sacrumFeatureIndices);
        database.addVectorXiItem("PelvisBoneMuscleTemplateData", "SacrumVertexIndices", sacrumVertexIndices);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TemplatePelvisCoarseShape", templatePelvisCoarseShape);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh", tempPelvisBoneMesh);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh", tempPelvisBoneMuscleMesh);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisMuscleMesh", tempPelvisMuscleMesh);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshIliumJoint", tempPelvisBoneMeshIliumJoint);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshLeftIlium", tempPelvisBoneMeshLeftIlium);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshLeftSacroiliacJoint", tempPelvisBoneMeshLeftSacroiliacJoint);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshLeftSacroiliacJointPickedPoints", tempPelvisBoneMeshLeftSacroiliacJointPickedPoints);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshPickedPoints", tempPelvisBoneMeshPickedPoints);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshLeftIliumPickedPoints", tempPelvisBoneMeshLeftIliumPickedPoints);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshRightIliumPickedPoints", tempPelvisBoneMeshRightIliumPickedPoints);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshSacrumPickedPoints", tempPelvisBoneMeshSacrumPickedPoints);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshPubicJoint", tempPelvisBoneMeshPubicJoint);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshPubicJointPickedPoints", tempPelvisBoneMeshPubicJointPickedPoints);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshRightIlium", tempPelvisBoneMeshRightIlium);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshRightSacroiliacJoint", tempPelvisBoneMeshRightSacroiliacJoint);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshRightSacroiliacJointPickedPoints", tempPelvisBoneMeshRightSacroiliacJointPickedPoints);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshSacrum", tempPelvisBoneMeshSacrum);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMeshWithOutJoints", tempPelvisBoneMeshWithOutJoints);
        database.addMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneShape", tempPelvisBoneShape);
        database.addMatrixXdItem("PelvisBoneMuscleTemplateData", "TempPelvisMusclesPickedPoints", tempPelvisMusclesPickedPoints);

        # Compacting the system file
        print("Compacting system file ...");
        database.compactSystemFile();
    addingTemplatePelvicBoneMuscleDataToTheSystemDatabase();

    # Finished processing
    print("Finished processing.");
def computePersonalizationQuality():
    # Initializing
    print("Initializing ...");
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    pelvicFolder = mainFolder + r"\FemalePelvisGeometries";
    personalizedPelvicFolder = mainFolder + r"\FemalePelvisGeometries\PersonalizedPelvisStructures";
    ctPelvicFolder = mainFolder + r"\FemalePelvisGeometries\AllPelvicStructures";
    templateFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    subjectIDFilePath = pelvicFolder + r"\FemalePelvisIDs.txt";

    # Reading initial information
    print("Reading initial information ...");
    pelvisIDs = sp.readListOfStrings(subjectIDFilePath);
    numOfSubjects = len(pelvisIDs);

    # Reading template information
    print("Reading template information ...");
    templatePelvicBoneMuscle = sp.readMesh(templateFolder + r"\TempPelvisBoneMuscles.ply");
    templatePelvicBone = sp.readMesh(templateFolder + r"\TempPelvisBoneMesh.ply");
    tempWithoutJointPelvicBone = sp.readMesh(templateFolder + r"\TempWithoutJointPelvisBoneMesh.ply");
    pelvicBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempWithoutJointPelvicBone.vertices, templatePelvicBoneMuscle.vertices);

    # Compute vertex to surface distances for all the subjects
    print("Computing vertex to surface distances ...");
    vertexDistances = [];
    for i in range(numOfSubjects):
        # Debugging
        print("Processing subject: ", i, " with ID: ", pelvisIDs[i]);
        subjectID = pelvisIDs[i];

        # Reading personalize pelvic bone mesh and the CT pelvic bone mesh
        personalizedPelvicBoneMuscle = sp.readMesh(personalizedPelvicFolder + f"/{subjectID}-PersonalizedPelvisBoneMuscleMesh.ply");
        personalizedPelvicBone = tempWithoutJointPelvicBone;
        personalizedPelvicBone.vertices = personalizedPelvicBoneMuscle.vertices[pelvicBoneVertexIndices];
        ctPelvicBone = sp.readMesh(ctPelvicFolder + f"/{subjectID}-PelvisBoneMesh.ply");

        # Compute the vertex to surface distances
        nearestCTVertices = sp.estimateNearestPointsFromPoints(personalizedPelvicBone.vertices, ctPelvicBone.vertices);
        distances = sp.computeCorrespondingDistancesPoints2Points(personalizedPelvicBone.vertices, nearestCTVertices);

        # Append the distances to data
        vertexDistances.append(distances);
    vertexDistances = np.array(vertexDistances);

    # Save the computed distances to file
    print("Saving computed distances to file ...");
    sp.saveNumPyArrayToNPY(pelvicFolder + f"/{subjectID}-VertexDistances.npy", vertexDistances);

    # Finished processing
    print("Finished processing.");
def drawPersonalizedErrorsOnTemplatePelvicBoneMesh():
    # Initialize
    print("Initializing ...");
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    pelvicFolder = mainFolder + r"\FemalePelvisGeometries";
    personalizedPelvicFolder = mainFolder + r"\FemalePelvisGeometries\PersonalizedPelvisStructures";
    ctPelvicFolder = mainFolder + r"\FemalePelvisGeometries\AllPelvicStructures";
    templateFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";

    # Load the template pelvic bone mesh
    print("Loading template pelvic bone mesh ...");
    templatePelvicBone = sp.readMesh(templateFolder + r"\TempWithoutJointPelvisBoneMesh.ply");

    # Loading vertex to surface distances 
    print("Loading vertex to surface distances ...");
    vertexDistances = sp.loadNumPYArrayFromNPY(pelvicFolder + f"/PersonalizedErrors.npy");

    # Draw the distance color maps on the template pelvic bone mesh
    print("Drawing distance color maps on the template pelvic bone mesh ...");
    ## Compute data for drawing
    meanTestingError = np.mean(vertexDistances, axis=0);
    ## Normalize the mean testing error to [0, 1] range
    normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
    ## Create a custom colormap from red (low) to violet (high)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
    colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
    ## Map the normalized mean testing error to colors
    colors = colormap(normalizedMeanTestingError);
    ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
    templatePelvicBone.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
    trimeshViewer = trimesh.Scene(templatePelvicBone);
    trimeshViewer.show();

    # Generate and display color ruler
    print("Generating color ruler ...");
    ## Create a figure for the color ruler
    fig, ax = plt.subplots(figsize=(8, 2))    
    ## Create a gradient for the colorbar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)    
    ## Display the gradient
    im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
    ## Calculate the actual error values for min, middle, and max
    min_error = np.min(meanTestingError) * 1000;  # Convert to mm
    max_error = np.max(meanTestingError) * 1000;  # Convert to mm
    mid_error = (min_error + max_error) / 2;
    ## Set up the colorbar ticks and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
    ax.set_yticks([])
    ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
    ax.set_title('Vertex-based Error Distribution between Personalized and CT-based Pelvic Bone Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
    ## Add grid lines for better readability
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
    ## Adjust layout and display
    plt.tight_layout()
    ## Save the color ruler figure
    plt.savefig(pelvicFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
    ## Show the color ruler
    plt.show()

    # Finished processing
    print("Finished processing.");
def trainThePCAModelForPersonalizedBoneMuscleMeshes():
    # Information
    # This procedure train the PCA model of the pelvic bone muscle shape model and analyze the trained model.

    # Initialization
    print("Initializing ...");
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    pelvicFolder = mainFolder + r"\FemalePelvisGeometries";
    personalizedPelvicFolder = mainFolder + r"\FemalePelvisGeometries\PersonalizedPelvisStructures";
    ctPelvicFolder = mainFolder + r"\FemalePelvisGeometries\AllPelvicStructures";
    templateFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";

    # Reading initial information
    print("Reading initial information ...");
    subjectIDFilePath = pelvicFolder + r"\FemalePelvisIDs.txt";
    pelvisIDs = sp.readListOfStrings(subjectIDFilePath);
    numOfSubjects = len(pelvisIDs);

    # Build the PCA statistical shape model of the pelvic bone muscle mesh
    print("Build the PCA statistical shape model ...");
    ## Forming the training data
    print("\t Forming the training data ...");
    trainingData = [];
    for i in range(numOfSubjects):
        # Debugging
        print(i, " ", end="", flush=True);

        # Load the personalized pelvic structure
        pelvicBoneMuscleMesh = sp.readMesh(personalizedPelvicFolder + f"/{pelvisIDs[i]}-PersonalizedPelvisBoneMuscleMesh.ply");

        # Forming the training data
        data = pelvicBoneMuscleMesh.vertices.flatten();
        trainingData.append(data);
    print("");
    ## Normalize the training data using available normalization technique
    print("\t Normalizing the training data ...");
    dataScaler = StandardScaler().fit(trainingData);
    scaledTrainingData = dataScaler.transform(trainingData);
    ## Train the PCA model
    print("\t Train the PCA model with 200 number of components ...");
    pca = PCA(n_components=200);
    pca.fit(scaledTrainingData);
    ## Transform the data to get all parameters
    print("\t Transform the data to get all parameters ...");
    transformedData = pca.transform(scaledTrainingData);
    ## Save all parameters to file
    print("\t Save all parameters to file ...");
    sp.saveNumPyArrayToNPY(pelvicFolder + "/PCA_TransformedData.npy", transformedData);
    ## Save the trained model to file
    print("\t Save the trained model to file ...");
    with open(pelvicFolder + "/PCA_Model.pkl", "wb") as f:
        pickle.dump(pca, f);
    ## Save the scaler to file
    print("\t Save the scaler to file ...");
    with open(pelvicFolder + "/DataScaler.pkl", "wb") as f:
        pickle.dump(dataScaler, f);

    # Finished processing
    print("Finished processing.");
def drawTheQualityOfTheTrainedPCAModel():
    # Initialize
    print("Initialize ...");
    def loadPCAModelFromPKL(file_path):
        """
        Load a PCA model from a pickle file.
        
        Parameters:
        - file_path (str): Path to the .pkl file containing the PCA model
        
        Returns:
        - PCA model object
        """
        try:
            with open(file_path, 'rb') as f:
                pcaModel = pickle.load(f)
            return pcaModel
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error loading PCA model: {e}")
            return None
    def loadDataScaler(filePath):
        """
        Load a data scaler from a pickle file.

        Parameters:
        - filePath (str): Path to the .pkl file containing the data scaler

        Returns:
        - Data scaler object
        """
        try:
            with open(filePath, 'rb') as f:
                dataScaler = pickle.load(f)
            return dataScaler
        except FileNotFoundError:
            print(f"File not found: {filePath}")
            return None
        except Exception as e:
            print(f"Error loading data scaler: {e}")
            return None
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    pelvicFolder = mainFolder + r"\FemalePelvisGeometries";
    shapeAnalyzingFolder = pelvicFolder + r"\ShapeVariationAnalyses";
    templateFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";

    # Load the PCA model
    print("Loading the PCA model ...");
    pcaModel = loadPCAModelFromPKL(pelvicFolder + "/PCA_Model.pkl");

    # Load the data scaler
    print("Load the data scaler ...");
    dataScaler = loadDataScaler(pelvicFolder + "/DataScaler.pkl");

    # Reading the transformed components
    print("Reading the transformed components ...");
    componentData = sp.loadNumPYArrayFromNPY(pelvicFolder + "/PCA_TransformedData.npy");

    # Reading the template pelvic bone muscle mesh
    print("Reading the template pelvic bone muscle mesh ...");
    templateMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMuscles.ply");

    # Draw the explained variance plot
    print("Drawing the explained variance plot ...");
    explained_variance_ratio = pcaModel.explained_variance_ratio_;
    # plt.figure(figsize=(12, 8));
    # plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'b-o', markersize=4);
    # plt.xlabel('Principal Component', fontweight='bold', fontsize=12);
    # plt.ylabel('Explained Variance Ratio', fontweight='bold', fontsize=12);
    # plt.title('Explained Variance Ratio per Component', fontweight='bold', fontsize=14);
    # plt.grid(True, alpha=0.3);
    # plt.xticks(fontweight='bold');
    # plt.yticks(fontweight='bold');
    # plt.show();

    # Draw the Cumulative variance plot
    print("Drawing the cumulative variance plot ...");
    cumulative_variance = np.cumsum(explained_variance_ratio);
    plt.figure(figsize=(12, 8));
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-o', markersize=4);
    plt.xlabel('Principal Component', fontweight='bold', fontsize=12);
    plt.ylabel('Cumulative Explained Variance', fontweight='bold', fontsize=12);
    plt.title('Cumulative Explained Variance per Component', fontweight='bold', fontsize=14);
    plt.grid(True, alpha=0.3);
    plt.xticks(fontweight='bold');
    plt.yticks(fontweight='bold');
    plt.show();

    # Drawing the shape variation visualization
    print("Drawing the shape variation visualization ...");
    # ## Get original number of components
    # originalNumComponents = pcaModel.n_components_;
    # ## Defining some functions for processing
    # def reducePCAComponents(pcaModel, newNumComponents):
    #     """
    #     Reduce the number of components in an existing PCA model.
        
    #     Parameters:
    #     - pcaModel: existing PCA model
    #     - newNumComponents: desired number of components
        
    #     Returns:
    #     - reduced PCA model
    #     """
    #     if newNumComponents > pcaModel.n_components_:
    #         print(f"Warning: Requested {newNumComponents} components, but model only has {pcaModel.n_components_}")
    #         return pcaModel
        
    #     # Create new PCA model with reduced components
    #     reducedPCA = PCA(n_components=newNumComponents)
        
    #     # Copy the relevant attributes from the original model
    #     reducedPCA.components_ = pcaModel.components_[:newNumComponents]
    #     reducedPCA.explained_variance_ = pcaModel.explained_variance_[:newNumComponents]
    #     reducedPCA.explained_variance_ratio_ = pcaModel.explained_variance_ratio_[:newNumComponents]
    #     reducedPCA.singular_values_ = pcaModel.singular_values_[:newNumComponents]
    #     reducedPCA.mean_ = pcaModel.mean_
    #     reducedPCA.n_components_ = newNumComponents
    #     reducedPCA.n_features_ = pcaModel.n_features_
    #     reducedPCA.n_samples_ = pcaModel.n_samples_
        
    #     return reducedPCA
    # def getOutputMeshFromPCA(pcaModel, dataScaler, componentValues, templateMesh):
    #     # Reduce the number of components based on the len of component values
    #     numComps = min(len(componentValues), pcaModel.n_components_);
    #     reducedPCA = reducePCAComponents(pcaModel, numComps);

    #     # Inverse the pca to form the mesh vertices
    #     scaledFlattenVertices = reducedPCA.inverse_transform(componentValues);
    #     scaledFlattenVertices = scaledFlattenVertices.reshape(1, -1);

    #     # Unscale the data
    #     flattenVertices = dataScaler.inverse_transform(scaledFlattenVertices);
    #     vertices = flattenVertices.reshape(-1, 3);

    #     # Forming the mesh
    #     outMesh = sp.cloneMesh(templateMesh);
    #     outMesh.vertices = vertices;

    #     # Return the mesh
    #     return outMesh;
    # ## Compute the mean shape
    # meanComponentValues = np.zeros(originalNumComponents);
    # meanMesh = getOutputMeshFromPCA(pcaModel, dataScaler, meanComponentValues, templateMesh);
    # ## Generate the mesh again with component data and render in the same trimesh scene
    # componentMeshes = [];
    # numOfMeshes = 10;
    # for i in range(numOfMeshes):
    #     # Get component values
    #     componentValues = componentData[i, :];
    #     # Get output of the mesh
    #     componentMesh = getOutputMeshFromPCA(pcaModel, dataScaler, componentValues, templateMesh);
    #     # Set the color of the mesh as rainbow color
    #     rgbColor = sp.generateRainBowColor(i, numOfMeshes);
    #     rgbColor = np.array(rgbColor)*255.0;
    #     # Set the grb color with alpha value
    #     componentMesh.visual.vertex_colors = np.array([rgbColor[0], rgbColor[1], rgbColor[2], 200], dtype=np.uint8);
    #     componentMeshes.append(componentMesh);
    # trimeshScene = trimesh.Scene([meanMesh] + componentMeshes);
    # trimeshScene.show();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_generateAugmentedAndTestIDs():
    # Initializing
    print("Initializing ...");
    disk = "H:";
    if (len(sys.argv) < 2):
        print("Please input the command as: [ProgramName] [ValidFold]"); return;
    validFold = int(sys.argv[1]);
    pelvisBoneReconFolder = disk + "/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisBoneReconFolder + "/CrossValidation";
    crossValidAugFolder = crossValidationFolder + f"/AugmentedTestingIDs/Fold_{validFold}"
    femalePelvisIDFilePath = pelvisBoneReconFolder + "/FemalePelvisIDs.txt"

    # Reading initial information
    print("Reading initial information ...");
    pelvisIDs = sp.readListOfStrings(femalePelvisIDFilePath);

    # Generate augmented IDs and testing IDs
    print("Generating augmented ids and testing ids ...");
    random.seed(42 + validFold);
    random.shuffle(pelvisIDs);
    splitRatio = 0.9
    splitIndex = int(len(pelvisIDs) * splitRatio);
    augmentedIds = pelvisIDs[:splitIndex];
    testingIds = pelvisIDs[splitIndex:];

    # Saving the augmented IDs and testing IDs
    print("Saving augmented IDs and testing IDs ...");
    sp.saveListOfStrings(crossValidAugFolder + "/AugmentedIDs.txt", augmentedIds);
    sp.saveListOfStrings(crossValidAugFolder + "/TestingIDs.txt", testingIds);

    # Generate ten times of training and validating data
    print("Generating ten times of training and validating data ...");
    numOfValids = 10; trainRatio = 0.8;
    for i in range(numOfValids):
        random.seed(i)  # Ensure reproducibility per run
        shuffledData = augmentedIds.copy();
        random.shuffle(shuffledData);

        splitIndex = int(len(shuffledData) * trainRatio);
        trainData = shuffledData[:splitIndex];
        validData = shuffledData[splitIndex:];

        sp.makeDirectory(crossValidAugFolder + "/TrainingValidIDs");

        sp.saveListOfStrings(crossValidAugFolder + f"/TrainingValidIDs/TrainingIDs_{i}.txt", trainData);
        sp.saveListOfStrings(crossValidAugFolder + f"/TrainingValidIDs/ValidIDs_{i}.txt", validData);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_augmentPelvisShapes_PCAGaussianMixture():
    # Information: 
    #  This procedure will generate the synthetic data on the training data and validating data for generatring the statistical shape model of the pelvis bone mesh
    #  the ssm of the pelvis bone mesh will be used for generating new data. The data will be trained and optimized on the augmented data. The optimized model will be
    #  tested on the tested data to see the efficiency of the data aumgnetaion.
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("Please input the command as: [ProgramName] [ValidFold]"); return;
    validFold = int(sys.argv[1]);
    disk = "F:";
    pelvisBoneReconFolder = disk + "/Data/PelvisBoneRecon";
    crossValidAugmentedFolder = pelvisBoneReconFolder + f"/CrossValidation/AugmentedTestingIDs/Fold_{validFold}";
    augmentingPelvisIDFilePath = crossValidAugmentedFolder + "/AugmentedIDs.txt"
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/";
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";

    # Initialize system database
    print("Initialize system data base ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    augmentingPelvisIDs = sp.readListOfStrings(augmentingPelvisIDFilePath);

    # Forming data for training
    print("Forming data for training ...");
    trainingData = [];
    for i, ID in enumerate(augmentingPelvisIDs):
        pelvisStructure = database.readMeshItem("PelvicStructureData", ID);
        pelvisVertices = pelvisStructure.vertices;
        trainingData.append(pelvisVertices.flatten());
    trainingData = np.array(trainingData);

    # Train the Gaussian mixture
    print("Training the Gaussian mixture ..."); 
    targetNumComps = trainingData.shape[0];
    pcaModel = PCA(n_components=targetNumComps);
    reduced_data = pcaModel.fit_transform(trainingData);
    gaussianMixtureModel = GaussianMixture(n_components=targetNumComps, covariance_type='full');
    gaussianMixtureModel.fit(reduced_data);
    dataAugmenter_PCAItemName = f"DataAugmenter_PCAModel_Fold_{validFold}";
    dataAugmenter_GaussianItemName = f"DataAugmenter_GaussianMixtureModel_Fold_{validFold}";

    # Save the data augmenter to the system database
    print("Save the data augmented to the system database ...")
    pelvisAugmenterDataFieldName = "PelvisStructureAugmenter";
    if (not database.fieldExists(pelvisAugmenterDataFieldName)): database.addField(pelvisAugmenterDataFieldName);
    if (not database.itemExists(pelvisAugmenterDataFieldName, dataAugmenter_PCAItemName)): 
        database.addPCAModelItem(pelvisAugmenterDataFieldName, dataAugmenter_PCAItemName, pcaModel);
    else: 
        database.updatePCAModelItem(pelvisAugmenterDataFieldName, dataAugmenter_PCAItemName, pcaModel);
    if (not database.itemExists(pelvisAugmenterDataFieldName, dataAugmenter_GaussianItemName)): 
        database.addGaussianMixtureModelItem(pelvisAugmenterDataFieldName, dataAugmenter_GaussianItemName, gaussianMixtureModel);
    else:
        database.updateGaussianMixtureModelItem(pelvisAugmenterDataFieldName, dataAugmenter_GaussianItemName, gaussianMixtureModel);
    
    # Sample to have the new subjects
    print("Sample to have new subject ...");
    targetNumOfSubjects = 1000;
    newReducedData = gaussianMixtureModel.sample(targetNumOfSubjects)[0];
    augmentedPelvisStructureFieldName = f"AugmentedPelvisStructures_Fold_{validFold}";
    if (not database.fieldExists(augmentedPelvisStructureFieldName)): database.addField(augmentedPelvisStructureFieldName);
    for i, newPelvisParams in enumerate(newReducedData):
        print(i, " ", flush=True, end="");
        newPelvisParams = newPelvisParams.reshape(1, -1);
        newPelvisVertices = pcaModel.inverse_transform(newPelvisParams);
        newPelvisVertices = newPelvisVertices.reshape(-1, 3);
        newPelvis = sp.cloneMesh(tempPelvisBoneMuscleMesh);
        newPelvis.vertices = newPelvisVertices;

        pelvisID = f"{i:04d}";
        if (not database.itemExists(augmentedPelvisStructureFieldName, pelvisID)):
            database.addMeshItem(augmentedPelvisStructureFieldName, pelvisID, newPelvis);
        else:
            database.updateMeshItem(augmentedPelvisStructureFieldName, pelvisID, newPelvis);
    
    # Finished processing
    print("Finished processing.");

#************************** CROSS-VALIDATION FUNCTIONS
#************* USING AFFINE TRANSFORM FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_affineTransform_BoneStructure():
    """
    Performs feature-to-pelvis structure reconstruction using affine transformation on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone mesh to match target feature points using rigid (SVD) and affine (CPD) transformations.
    The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Deform template mesh to match target features using transformations
       - Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    """

    # Initialize command line argument parsing and validate input parameters
    # Expects: [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]
    
    # Setup file paths and database connections
    # - System database for storing pelvis structure data
    # - Cross-validation folders for training/validation splits
    # - Feature selection protocol definitions
    # - Output directory for storing computed errors
    
    # Load template pelvis bone mesh and compute vertex mapping indices
    # Template serves as the base mesh that will be deformed to match targets
    
    # Load all available feature selection strategies from protocol file
    # Each strategy defines a subset of anatomical feature points to use
    
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/AffineDeformation/BoneStructures";
    
    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneVertexData = np.array(validPelvisBoneVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneVertices = validPelvisBoneVertexData[v].reshape(-1, 3);
                validPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                validPelvisBoneMesh.vertices = validPelvisBoneVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(defPelvisBoneMesh.vertices, validPelvisBoneMesh.vertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_affineTransform_BoneMuscleStructure():
    """
    Performs feature-to-pelvis structure reconstruction using affine transformation on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone mesh to match target feature points using rigid (SVD) and affine (CPD) transformations.
    The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Deform template mesh to match target features using transformations
       - Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5"; 
    outFolder = crossValidationFolder + "/AffineDeformation/BoneMuscleStructures";
    
    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneMuscleVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvisBoneMuscleMesh = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneMuscleVertexData.append(pelvisBoneMuscleMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneMuscleVertexData = np.array(validPelvisBoneMuscleVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertexData[v].reshape(-1, 3);
                validPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                validPelvisBoneMuscleMesh.vertices = validPelvisBoneMuscleVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);
                
                # Compute only the bone vertices for computing distances
                validPelvisBoneVertices = validPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
                personalizedBoneVertices = defPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(personalizedBoneVertices, validPelvisBoneVertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");

#************* USING RADIAL BASIS FUNCTION INTERPOLATION FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_radialBasicFunctionInterpolation_BoneStructures():
    """
    Performs feature-to-pelvis structure reconstruction using radial basis function interpolation on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone mesh to match target feature points using rigid (SVD), affine (CPD), and radial basis function 
    transformations. The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Apply rigid SVD transformation for initial alignment
       - Apply affine CPD transformation for further refinement
       - Apply radial basis function interpolation for final deformation
       - Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    - Radial Basis Function (RBF): Non-rigid deformation using thin plate splines
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneStructures";
    
    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneVertexData = np.array(validPelvisBoneVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneVertices = validPelvisBoneVertexData[v].reshape(-1, 3);
                validPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                validPelvisBoneMesh.vertices = validPelvisBoneVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMesh = sp.transformMesh(defPelvisBoneMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);

                # Deform using the radial basic function
                deformedToPersonalizedPelvisFeatureDisplacements =  targetFeaturePoints - defFeaturePoints;
                rbf = RBFInterpolator(defFeaturePoints, deformedToPersonalizedPelvisFeatureDisplacements, kernel='thin_plate_spline');
                personalizedPelvisBoneVertexDisplacements = rbf(defPelvisBoneMesh.vertices);
                personalizedPelvisBoneMesh = sp.cloneMesh(defPelvisBoneMesh);
                personalizedPelvisBoneMesh.vertices = defPelvisBoneMesh.vertices + personalizedPelvisBoneVertexDisplacements;

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(personalizedPelvisBoneMesh.vertices, validPelvisBoneMesh.vertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_radialBasicFunctionInterpolation_BoneMuscleStructures():
    """
    Performs feature-to-pelvis structure reconstruction using radial basis function interpolation on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by deforming a template pelvis 
    bone-muscle mesh to match target feature points using rigid (SVD), affine (CPD), and radial basis function 
    transformations. The evaluation is performed across multiple feature selection strategies and cross-validation folds.
    
    Key Steps:
    1. Parse command line arguments for processing range
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Apply rigid SVD transformation for initial alignment
       - Apply affine CPD transformation for further refinement
       - Apply radial basis function interpolation for final deformation
       - Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Transformations Applied:
    - Rigid SVD transformation: Aligns template to target via rotation/translation
    - Affine CPD transformation: Further refines alignment with scaling/shearing
    - Radial Basis Function (RBF): Non-rigid deformation using thin plate splines
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    
    Note: Accuracy is computed only on bone vertices since muscle ground truth may not be available
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5"; 
    outFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneMuscleStructures";
    
    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvisFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            validPelvisBoneMuscleVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(validIDs):
                pelvisBoneMuscleMesh = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneMuscleVertexData.append(pelvisBoneMuscleMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            validPelvisBoneMuscleVertexData = np.array(validPelvisBoneMuscleVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Try to deform the template pelvis bone mesh to the target pelvis bone features
            print("\t Deforming template pelvis bone mesh to target pelvis bone features ...");
            avgP2PDists = [];
            for v, validPelvisFeatures in enumerate(validPelvisFeatureData):
                # Debugging
                print("#********************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, " VALID SUBJECT INDEX: ", v);

                # Getting target information
                targetFeaturePoints = validPelvisFeatures.reshape(-1, 3);
                validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertexData[v].reshape(-1, 3);
                validPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                validPelvisBoneMuscleMesh.vertices = validPelvisBoneMuscleVertices;

                # Deform the template pelvis bone to the target feature points
                defPelvisBoneMuscleMesh = sp.cloneMesh(tempPelvisBoneMuscleMesh);
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(defPelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                svdTransform = sp.estimateRigidSVDTransform(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, svdTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, svdTransform);
                affineTransform = sp.estimateAffineTransformCPD(defFeaturePoints, targetFeaturePoints);
                defPelvisBoneMuscleMesh = sp.transformMesh(defPelvisBoneMuscleMesh, affineTransform);
                defFeaturePoints = sp.transform3DPoints(defFeaturePoints, affineTransform);
                
                # Deform using the radial basic function
                deformedToPersonalizedPelvisFeatureDisplacements =  targetFeaturePoints - defFeaturePoints;
                rbf = RBFInterpolator(defFeaturePoints, deformedToPersonalizedPelvisFeatureDisplacements, kernel='thin_plate_spline');
                personalizedPelvisBoneMuscleVertexDisplacements = rbf(defPelvisBoneMuscleMesh.vertices);
                personalizedPelvisBoneMuscleMesh = sp.cloneMesh(defPelvisBoneMuscleMesh);
                personalizedPelvisBoneMuscleMesh.vertices = defPelvisBoneMuscleMesh.vertices + personalizedPelvisBoneMuscleVertexDisplacements;
                
                defFeaturePoints = sp.reconstructLandmarksFromBarycentric(personalizedPelvisBoneMuscleMesh, pelvisFeatureBaryIndices, pelvicFeatureBaryCoords);
                
                # Compute only the bone vertices for computing distances
                validPelvisBoneVertices = validPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
                personalizedBoneVertices = personalizedPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

                # Computing points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, personalizedBoneVertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors to file
            sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");

#************* USING SHAPE OPTIMIZATION STRATEGY FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_shapeOptimizationStrategy_BoneStructures():
    """
    Performs feature-to-pelvis structure reconstruction using shape optimization strategy on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by optimizing a statistical shape model 
    (SSM) of pelvis bones to match target feature points using least squares optimization.
    The evaluation is performed across multiple feature selection strategies, cross-validation folds, and 
    varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Train PCA statistical shape model on training data with specified components
       - For each validation subject:
         * Initialize parameters from mean training data
         * Optimize SSM parameters to minimize feature point distances using least squares
         * Reconstruct pelvis bone mesh from optimized parameters
         * Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Optimization Strategy:
    - PCA Statistical Shape Model: Learns shape variations from training data
    - Least Squares Optimization: Minimizes distance between reconstructed and target feature points
    - Parameter Space Search: Optimizes in reduced PCA parameter space
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneStructures";
    def reconstructPelvisBoneMesh(pcaModel, paramData, meshFaces):
        """Reconstruct pelvis bone mesh vertices from PCA model."""
        meshVertices = pcaModel.inverse_transform(paramData);
        meshVertices = meshVertices.reshape(-1, 3);
        outMesh = sp.formMesh(meshVertices, meshFaces);
        return outMesh;
    def computeFeaturePoints(pelvisBoneMesh, baryIndices, baryCoords):
        """Compute feature points using barycentric interpolation."""
        featurePoints = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, baryIndices, baryCoords);
        return featurePoints
    def optimizePelvisBoneSSM(initialParams, pcaModel, meshFaces, baryIndices, baryCoords, targetFeaturePoints):
        """Optimization function to align pelvis bone feature points with target points."""
        reconstructedMesh = reconstructPelvisBoneMesh(pcaModel, initialParams.reshape(1, -1), meshFaces)
        computedFeaturePoints = computeFeaturePoints(reconstructedMesh, baryIndices, baryCoords)
        return (computedFeaturePoints - targetFeaturePoints).flatten();

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainPelvisBoneVertexData = []; 
            validPelvisBoneVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                trainPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            trainPelvisBoneVertexData = np.array(trainPelvisBoneVertexData);
            validPelvisBoneVertexData = np.array(validPelvisBoneVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Train the statistial shape model of the pelvis bone structure
                pelvisBoneNumComps = min(numComps, trainPelvisBoneVertexData.shape[1]);
                pelvisBoneSSM = PCA(n_components=pelvisBoneNumComps);
                pelvisBoneSSM.fit(trainPelvisBoneVertexData);

                # Optimize the trained PCA model to the valid features
                avgP2PDists = [];
                for v, validFeatures in enumerate(validPelvisFeatureData):
                    # Getting validating feature points
                    validFeaturePoints = validFeatures.reshape(-1, 3);
                    validPelvisBoneVertices = validPelvisBoneVertexData[v];
                    validPelvisBoneVertices = validPelvisBoneVertices.reshape(-1, 3);

                    # Optimize to the target features
                    initialParams = pelvisBoneSSM.transform(trainPelvisBoneVertexData.mean(axis=0).reshape(1, -1)).flatten();
                    result = least_squares(optimizePelvisBoneSSM, initialParams, args=(pelvisBoneSSM, tempPelvisBoneMesh.faces, 
                                                                                       pelvicFeatureBaryIndices, pelvicFeatureBaryCoords, 
                                                                                       validFeaturePoints), verbose=0, ftol=1e-12);
                    optimizedParams = result.x.reshape(1, -1);
                    optimizedPelvisBoneVertices = pelvisBoneSSM.inverse_transform(optimizedParams);
                    optimizedPelvisBoneVertices = optimizedPelvisBoneVertices.reshape(-1, 3);

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(optimizedPelvisBoneVertices, validPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDist = np.array(avgP2PDist);

                # Save the comptued errors to file
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeOptimizationStrategy_BoneMuscleStructures():
    """
    Performs feature-to-pelvis structure reconstruction using shape optimization strategy on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by optimizing a statistical shape model 
    (SSM) of pelvis bone-muscle structures to match target feature points using least squares optimization.
    The evaluation is performed across multiple feature selection strategies, cross-validation folds, and 
    varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Train PCA statistical shape model on training data with specified components
       - For each validation subject:
         * Initialize parameters from mean training data
         * Optimize SSM parameters to minimize feature point distances using least squares
         * Reconstruct pelvis bone-muscle mesh from optimized parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Optimization Strategy:
    - PCA Statistical Shape Model: Learns shape variations from training data
    - Least Squares Optimization: Minimizes distance between reconstructed and target feature points
    - Parameter Space Search: Optimizes in reduced PCA parameter space
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Accuracy is computed only on bone vertices since muscle ground truth may not be available
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";
    outFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneMuscleStructures";
    def reconstructPelvisBoneMesh(pcaModel, paramData, meshFaces):
        """Reconstruct pelvis bone mesh vertices from PCA model."""
        meshVertices = pcaModel.inverse_transform(paramData);
        meshVertices = meshVertices.reshape(-1, 3);
        outMesh = sp.formMesh(meshVertices, meshFaces);
        return outMesh;
    def computeFeaturePoints(pelvisBoneMesh, baryIndices, baryCoords):
        """Compute feature points using barycentric interpolation."""
        featurePoints = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, baryIndices, baryCoords);
        return featurePoints
    def optimizePelvisBoneSSM(initialParams, pcaModel, meshFaces, baryIndices, baryCoords, targetFeaturePoints):
        """Optimization function to align pelvis bone feature points with target points."""
        reconstructedMesh = reconstructPelvisBoneMesh(pcaModel, initialParams.reshape(1, -1), meshFaces)
        computedFeaturePoints = computeFeaturePoints(reconstructedMesh, baryIndices, baryCoords)
        return (computedFeaturePoints - targetFeaturePoints).flatten();

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainPelvisBoneMuscleVertexData = []; 
            validPelvisBoneMuscleVertexData = []; validPelvisFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                trainPelvisBoneMuscleVertexData.append(pelvicStructure.vertices.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvisBoneMuscleVertexData.append(pelvicStructure.vertices.flatten());
                validPelvisFeatureData.append(pelvicFeatures.flatten());
            trainPelvisBoneMuscleVertexData = np.array(trainPelvisBoneMuscleVertexData);
            validPelvisBoneMuscleVertexData = np.array(validPelvisBoneMuscleVertexData);
            validPelvisFeatureData = np.array(validPelvisFeatureData);

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Train the statistial shape model of the pelvis bone structure
                pelvisBoneMuscleNumComps = min(numComps, trainPelvisBoneMuscleVertexData.shape[1]);
                pelvisBoneMuscleSSM = PCA(n_components=pelvisBoneMuscleNumComps);
                pelvisBoneMuscleSSM.fit(trainPelvisBoneMuscleVertexData);

                # Optimize the trained PCA model to the valid features
                avgP2PDists = [];
                for v, validFeatures in enumerate(validPelvisFeatureData):
                    # Getting validating feature points
                    validFeaturePoints = validFeatures.reshape(-1, 3);
                    validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertexData[v];
                    validPelvisBoneMuscleVertices = validPelvisBoneMuscleVertices.reshape(-1, 3);

                    # Optimize to the target features
                    initialParams = pelvisBoneMuscleSSM.transform(trainPelvisBoneMuscleVertexData.mean(axis=0).reshape(1, -1)).flatten();
                    result = least_squares(optimizePelvisBoneSSM, initialParams, args=(pelvisBoneMuscleSSM, tempPelvisBoneMuscleMesh.faces, 
                                                                                       pelvicFeatureBaryIndices, pelvicFeatureBaryCoords, 
                                                                                       validFeaturePoints), verbose=0, ftol=1e-12);
                    optimizedParams = result.x.reshape(1, -1);
                    optimizedPelvisBoneMuscleVertices = pelvisBoneMuscleSSM.inverse_transform(optimizedParams);
                    optimizedPelvisBoneMuscleVertices = optimizedPelvisBoneMuscleVertices.reshape(-1, 3);

                    # Compute only accuracy on the pelvis bone structure
                    validPelvisBoneVertices = validPelvisBoneMuscleVertices[pelvisBoneVertexIndices];
                    optimPelvisBoneVertices = optimizedPelvisBoneMuscleVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(optimPelvisBoneVertices, validPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDist = np.array(avgP2PDist);

                # Save the comptued errors to file
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");

#************* USING SHAPE RELATION STRATEGY FOR BONE AND BONE MUSCLE STRUCTURES
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneStructures():
    """
    Performs feature-to-pelvis structure reconstruction using shape relation strategy on bone structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone shape parameters using PCA dimensionality reduction and linear regression.
    The evaluation is performed across multiple feature selection strategies, cross-validation folds, and 
    varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train linear regression model to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone mesh from predicted parameters
         * Compute point-to-point distance errors
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Linear Regression: Learns mapping between feature and vertex parameter spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "G:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneStructures";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);
    
    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainPelvisBoneVertexData = []; validPelvicVertexData = [];
            trainPelvisFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainPelvisBoneVertexData.append(pelvisBoneMesh.vertices.flatten());
                trainPelvisFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvisBoneMesh = sp.cloneMesh(tempPelvisBoneMesh);
                pelvisBoneMesh.vertices = pelvicStructure.vertices[pelvisBoneVertexIndices];
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvisBoneMesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvisBoneMesh.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                print("\t Scaling data ...");
                yScaler = StandardScaler().fit(trainPelvisBoneVertexData);
                xScaler = StandardScaler().fit(trainPelvisFeatureData);
                scaledYData = yScaler.transform(trainPelvisBoneVertexData);
                scaledXData = xScaler.transform(trainPelvisFeatureData);

                # Control the number of components
                print("\t Controlling the number of components ...");
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                print("\t Parameterization training ...");
                print("\t\t Parameterizing the scaledXData ...");
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                print("\t\t Parameterizing the scaledYData ...");
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                print("\t Regression training ...");
                regressor = LinearRegression()
                regressor.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                print("\t Computing valiation errors ...");
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = regressor.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(predPelvicStructureVertices, validPelvicStructureVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                print("\t Save the computed errors ...");
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_multivarirateLinearRegression():
    """
    Performs feature-to-pelvis structure reconstruction using multivariate linear regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and multivariate 
    linear regression. The evaluation is performed across multiple feature selection strategies, cross-validation 
    folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train multivariate linear regression model to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Multivariate Linear Regression: Learns direct mapping between feature and vertex parameter spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Accuracy is computed only on bone vertices since muscle ground truth may not be available
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                regressor = LinearRegression()
                regressor.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = regressor.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_RidgeLinearRegression():
    """
    Performs feature-to-pelvis structure reconstruction using ridge linear regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and ridge 
    linear regression with L2 regularization. The evaluation is performed across multiple feature selection 
    strategies, cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train ridge linear regression model with L2 regularization to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained ridge regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Ridge Linear Regression: Learns mapping with L2 regularization to prevent overfitting
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Ridge regression adds L2 penalty to prevent overfitting compared to standard linear regression.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_RidgeLinearRegression";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                regressor = Ridge(alpha=1.0);
                regressor.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = regressor.predict(validXParams);                
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_CanonicalCorrelationAnalysis():
    """
    Performs feature-to-pelvis structure reconstruction using canonical correlation analysis on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and canonical 
    correlation analysis (CCA). The evaluation is performed across multiple feature selection strategies, 
    cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train canonical correlation analysis model to find correlated components
       - Train linear regression on canonical components to map feature to vertex spaces
       - For each validation subject:
         * Transform features to CCA canonical space
         * Predict vertex canonical components using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Canonical Correlation Analysis: Finds maximally correlated linear combinations between spaces
    - Linear Regression: Maps between canonical components of feature and vertex spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: CCA finds linear combinations that maximize correlation between feature and vertex spaces.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";    
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_CanonicalCorrelationAnalysis";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);
                targetNumComps = min(xNumComps, yNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=targetNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=targetNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                cca = CCA(n_components=targetNumComps);
                cca.fit(trainXParamData, trainYParamData);
                X_c, Y_c = cca.transform(trainXParamData, trainYParamData);
                reg = LinearRegression();
                reg.fit(X_c, Y_c);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                X_valid_c = cca.transform(validXParams);
                Y_pred_c = reg.predict(X_valid_c);
                predYParams = cca.inverse_transform(Y_pred_c);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);                
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_PartialLeastSquaresRegression():
    """
    Performs feature-to-pelvis structure reconstruction using partial least squares regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and partial 
    least squares (PLS) regression. The evaluation is performed across multiple feature selection strategies, 
    cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train partial least squares regression model to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained PLS regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Partial Least Squares Regression: Finds linear combinations that maximize covariance between spaces
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: PLS regression maximizes covariance between feature and vertex parameter spaces.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_PartialLeastSquaresRegression";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);
                targetNumComps = min(xNumComps, yNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=targetNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=targetNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                pls = PLSRegression(n_components=targetNumComps);
                pls.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = pls.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predScaledYData = ySSM.inverse_transform(predYParams);                
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_GaussianProcessRegressor():
    """
    Performs feature-to-pelvis structure reconstruction using Gaussian process regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and Gaussian 
    process regression with RBF kernel. The evaluation is performed across multiple feature selection strategies, 
    cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train Gaussian process regressor with RBF kernel to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained Gaussian process model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Gaussian Process Regression: Non-parametric Bayesian approach with RBF kernel
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Gaussian processes provide uncertainty quantification and handle non-linear relationships.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_GaussianProcessRegressor";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                gpr = GaussianProcessRegressor(kernel=RBF())
                gpr.fit(trainXParamData, trainYParamData);

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = gpr.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predYParams = predYParams.reshape(-1, yNumComps);
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor():
    """
    Performs feature-to-pelvis structure reconstruction using multi-output ridge regression on bone-muscle structures.
    
    This function evaluates the accuracy of pelvis bone reconstruction by learning statistical relationships 
    between feature points and bone-muscle shape parameters using PCA dimensionality reduction and multi-output 
    ridge regression with cross-validation. The evaluation is performed across multiple feature selection 
    strategies, cross-validation folds, and varying numbers of principal components.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize system database and load template data
    3. Load feature selection strategies 
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train multi-output ridge regression model with cross-validation to map feature parameters to vertex parameters
       - For each validation subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Shape Relation Strategy:
    - PCA Feature Space: Reduces feature dimensionality while preserving variance
    - PCA Vertex Space: Creates compact representation of shape variations
    - Multi-Output Ridge Regression: Uses RidgeCV for automatic hyperparameter tuning across multiple outputs
    - Cross-validation: Ensures robust evaluation across different data splits
    
    Output:
    - CSV files containing average point-to-point distances for each feature strategy, 
      validation fold, and number of components configuration
    
    Note: Multi-output regression handles multiple target variables simultaneously with built-in cross-validation.
    Accuracy is computed only on bone vertices since muscle ground truth may not be available.
    """
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatSelStratIndex] [EndFeatSelStratIndex] [StartValidIndex] [EndValidIndex] [StartNumComps] [EndNumComps]");
        return;
    startFeatSelStratIndex = int(sys.argv[1]); endFeatSelStratIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);
    disk = "H:";
    debugFolder = disk + r"\Data\PelvisBoneRecon\Debugs";
    systemDatabaseFolder = disk + "/Data/PelvisBoneRecon/SystemDataBase/"
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    trainValidTestIDFolder = crossValidationFolder + "/TrainingValidTestingIDs";
    featureSelectionProtocolFolder = crossValidationFolder + "/FeatureSelectionProtocol";    
    systemDatabaseFilePath = systemDatabaseFolder + "/SystemDatabase.h5";
    outFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_MultiOutputRegressor";

    # Initialize databse
    print("Initialize database ...");
    database = SystemDatabaseManager(systemDatabaseFilePath);

    # Reading initial information
    print("Reading initial information ...");
    tempPelvisBoneMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMesh");
    tempPelvisBoneMuscleMesh = database.readMeshItem("PelvisBoneMuscleTemplateData", "TempPelvisBoneMuscleMesh");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(tempPelvisBoneMesh.vertices, tempPelvisBoneMuscleMesh.vertices);

    # Reading feature section strategies
    print("Reading feature selection strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");    
    
    # Iterate for each feature selection strategy
    print("Iterate for each feature selection strategy ...");
    for featSelIndex in range(startFeatSelStratIndex, endFeatSelStratIndex + 1):
        # Debugging
        print("/********************************************** FEATURE SELECTION STRATEGY: ", featSelIndex);

        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(tempPelvisBoneMuscleMesh, tempPelvicFeatures);

        # Conduct training validating throughout each time of cross-validation
        for validIndex in range(startValidIndex, endValidIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainValidTestIDFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = database.readMeshItem("PelvicStructureData", ID);
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());

            # Begin to train for each number of components
            print("\t Begin to train in each number of components ...");
            for numComps in range(startNumComps, endNumComps + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                model = MultiOutputRegressor(RidgeCV())
                model.fit(trainXParamData, trainYParamData)

                # Computing errors on the validating data
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = model.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predYParams = predYParams.reshape(-1, yNumComps);
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                sp.saveMatrixToCSVFile(outFolder + f"/AveragePoint2PointDistances_{featSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_selectOptimalReconstructionStrategy():
    """
    Analyzes and compares the performance of different feature-to-pelvis reconstruction strategies to identify the optimal approach.
    
    This function evaluates multiple reconstruction methodologies by loading validation/testing error data 
    and generating comparative visualizations to determine the best performing strategy for pelvis structure 
    reconstruction from anatomical feature points.
    
    Key Analysis Components:
    1. Affine Transform-based Non-rigid Deformation
    2. Radial Basis Function Interpolation-based Non-rigid Deformation  
    3. Statistical Shape Optimization
    4. Statistical Shape Relation with various regression models
    5. Performance comparison across different feature selection strategies
    
    Reconstruction Strategies Evaluated:
    - Affine Transform: Uses rigid SVD + affine CPD transformations
    - RBF Interpolation: Adds radial basis function deformation for non-rigid alignment
    - Shape Optimization: PCA-based statistical shape model with least squares optimization
    - Shape Relation: PCA + regression mapping between feature and shape parameter spaces
    
    Regression Models Analyzed:
    - Multivariate Linear Regression
    - Ridge Linear Regression
    - Canonical Correlation Analysis
    - Partial Least Squares Regression
    - Gaussian Process Regression
    - Multi-output Ridge Regression
    
    Visualization Outputs:
    - Box plots showing error distributions for each method
    - Line plots showing optimal number of components vs validation errors
    - Bar charts comparing mean errors across different strategies
    - Statistical trend analysis with polynomial fitting
    
    Output:
    - Comparative performance charts identifying the optimal reconstruction strategy
    - Error trend analysis across different numbers of feature points
    - Statistical summaries for method selection guidance
    
    Note: This function serves as a comprehensive evaluation tool to guide the selection
    of the most effective feature-to-pelvis reconstruction methodology based on empirical
    validation results across multiple cross-validation folds.
    """

    # Initializing
    print("Initializing ...");
    disk = "H:";
    crossValidationFolder = disk + r"/Data/PelvisBoneRecon/CrossValidation";
    featureSelIndex = 10;
    optimNumCompBoneOptim = 18;
    optimNumCompBoneMuscleOptim = 18;

    # Define error drawing function
    def affineTransformNonRigidDeformation():
        """
        Analyzes and visualizes the performance of affine transform-based non-rigid deformation strategy.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        affine transformation methods by loading validation error data and generating statistical 
        visualizations to assess the effectiveness of this approach.
        
        Key Steps:
        1. Load validation error data from affine transform bone-muscle structure results
        2. Aggregate mean validation errors across all cross-validation folds
        3. Convert error measurements to millimeters for presentation
        4. Generate box plot visualization showing error distribution
        5. Annotate plot with mean error values and statistical information
        
        Methodology Analysis:
        - Affine Transform Strategy: Uses rigid SVD transformation followed by affine CPD transformation
        - Validation Framework: Processes errors across multiple cross-validation folds (0-9)
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Box plot showing error distribution for pelvic structure reconstruction
        - Mean error annotation with precise measurements
        - Professional formatting with bold fonts and clear labeling
        - Title indicating the specific deformation strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual assessment of method performance and variability
        
        Note: This analysis focuses specifically on bone-muscle structure reconstruction
        using affine transformation as the primary deformation strategy, providing
        baseline performance metrics for comparison with other reconstruction approaches.
        """
        # Initializing
        print("\t Initializing ...");
        validErrorFolder = crossValidationFolder + "/AffineDeformation/BoneMuscleStructures";

        # Reading validation errors for bone and muscle structure
        print("\t Reading validating errors for bone and muscle structures ...");
        boneMuscleStructureValidErrorData = [];    
        minValidIndex = 0; maxValidIndex = 9;
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(validErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            boneMuscleStructureValidErrorData.append(meanValidError);
        boneMuscleStructureValidErrorData = np.array(boneMuscleStructureValidErrorData);
        
        print("\t Drawing box plots for errors distribution ...")
        # Convert errors to millimeters
        drawingData = boneMuscleStructureValidErrorData * 1000
        # Create box plot
        plt.boxplot([drawingData],
                    labels=["Pelvic Structure"],
                    medianprops=dict(color="red"))
        # Set title with bold and larger font
        plt.title("Affine Transform-based\nNon-rigid Deformation",
                fontsize=16, fontweight='bold')
        # Set y-axis label
        plt.ylabel("Validation Errors (mm)", fontsize=14, fontweight='bold')
        # Set x-axis label
        plt.xticks(fontsize=12, fontweight='bold')
        # Set y-ticks (index numbers) font size and weight
        plt.yticks(fontsize=12, fontweight='bold')
        # No grid
        plt.grid(False)
        # Add mean values
        mean2 = np.mean(drawingData)
        # Annotate means (adjust position as needed)
        plt.text(1, mean2, f"{mean2:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
        plt.show()
    def radialBasisFunctionInterpolation():
        """
        Analyzes and visualizes the performance of radial basis function interpolation-based non-rigid deformation strategy.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        radial basis function interpolation methods by loading validation error data and generating statistical 
        visualizations to assess the effectiveness of this approach.
        
        Key Steps:
        1. Load validation error data from radial basis function bone-muscle structure results
        2. Aggregate mean validation errors across all cross-validation folds
        3. Convert error measurements to millimeters for presentation
        4. Generate box plot visualization showing error distribution
        5. Annotate plot with mean error values and statistical information
        
        Methodology Analysis:
        - RBF Interpolation Strategy: Uses rigid SVD + affine CPD + radial basis function transformations
        - Validation Framework: Processes errors across multiple cross-validation folds (0-9)
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Box plot showing error distribution for pelvic structure reconstruction
        - Mean error annotation with precise measurements
        - Professional formatting with bold fonts and clear labeling
        - Title indicating the specific deformation strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual assessment of method performance and variability
        
        Note: This analysis focuses specifically on bone-muscle structure reconstruction
        using radial basis function interpolation as the primary deformation strategy, providing
        enhanced non-rigid deformation capabilities compared to affine transformation alone.
        """
            
        # Initializing
        print("\t Initializing ...");
        boneStructureValidErrorFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneStructures";
        boneMuscleStructureValidErrorFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneMuscleStructures";

        # Reading validation errors for bone and muscle structure
        print("\t Reading validating errors for bone and muscle structures ...");
        validErrorData = [];
        minValidIndex = 0; maxValidIndex = 9;
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(boneMuscleStructureValidErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            validErrorData.append(meanValidError);
        validErrorData = np.array(validErrorData);
        
        print("\t Drawing box plots for errors distribution ...")
        # Convert errors to millimeters
        drawingData = validErrorData * 1000
        # Create box plot
        plt.boxplot([drawingData],
                    labels=["Pelvic Structure"],
                    medianprops=dict(color="red"))
        # Set title with bold and larger font
        plt.title("Radial Basis Function Interpolation-based\nNon-rigid Deformation",
                fontsize=16, fontweight='bold')
        # Set y-axis label
        plt.ylabel("Validation Errors (mm)", fontsize=14, fontweight='bold')
        # Set x-axis label
        plt.xticks(fontsize=12, fontweight='bold')
        # Set y-ticks (index numbers) font size and weight
        plt.yticks(fontsize=12, fontweight='bold')
        # No grid
        plt.grid(False)
        # Add mean values
        mean2 = np.mean(drawingData)
        # Annotate means (adjust position as needed)
        plt.text(1, mean2, f"{mean2:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
        plt.show()
    def statisticalShapeOptimization():
        """
        Analyzes and visualizes the performance of statistical shape optimization strategy for pelvis reconstruction.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        statistical shape optimization methods by loading validation error data across varying numbers 
        of principal components and generating performance visualizations.
        
        Key Steps:
        1. Load validation error data from shape optimization bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Optimization Strategy: Uses PCA-based statistical shape models with least squares optimization
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific optimization strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: This analysis focuses on statistical shape optimization using PCA-based models
        to determine the optimal balance between model complexity and reconstruction accuracy.
        """
        # Initializing
        print("\t Initializing ...");
        validErrorFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneMuscleStructures";
    
        print("\t Reading valiating errors for bone and muscle structures ...");
        maxNumComps = 80; minNumComps = 1;
        minValidIndex = 0; maxValidIndex = 9;
        validErrorData = np.zeros((10, maxNumComps));         
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(validErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validErrorData[validIndex,  numComps - 1] = validError;
        validErrorData = np.array(validErrorData);
        
        # Drawing the graphs
        print("\t Drawing the graph ...");
        drawingData = np.mean(validErrorData, axis=0) * 1000
        N = validErrorData.shape[1]
        x = np.arange(1, N + 1)
        y = drawingData

        # Find minimum points
        minIndex = np.argmin(y)
        minX = x[minIndex]
        minY = y[minIndex]

        # Plotting the curves
        line2, = plt.plot(x, y, label=f'Pelvic Structure (min: {minX}, {minY:.2f} mm)')

        # Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Statistical Shape Optimization', fontdict={'fontsize': 16, 'fontweight': 'bold'})
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial')
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial')

        # Mark and annotate minimum points
        plt.scatter(minX, minY, color=line2.get_color(), zorder=3)

        # Add legend in top-left
        plt.legend(loc='upper left', fontsize=12, frameon=False)

        plt.show()
    def statisticalShapeRelation():
        """
        Analyzes and visualizes the performance of statistical shape relation strategy for pelvis reconstruction.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        statistical shape relation methods by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from shape relation bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific relation strategy being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: This analysis focuses on statistical shape relation using regression-based models
        to determine the optimal balance between feature dimensionality and reconstruction accuracy.
        """
        # Initializing
        print("\t Initializing ...");
        validErrorFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
    
        print("\t Reading valiating errors for bone and muscle structures ...");
        maxNumComps = 80; minNumComps = 1;
        minValidIndex = 0; maxValidIndex = 9;
        validErrorData = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(validErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validErrorData[validIndex,  numComps - 1] = validError;
        validErrorData = np.array(validErrorData);

        # Drawing the graphs
        print("\t Drawing the graph ...");
        drawingData2 = np.mean(validErrorData, axis=0) * 1000
        N = validErrorData.shape[1]
        x = np.arange(1, N + 1)
        y = drawingData2

        # Find minimum points
        minIndex = np.argmin(y)
        minX = x[minIndex]
        minY = y[minIndex]

        # Plotting the curves
        line2, = plt.plot(x, y, label=f'Pelvic Structure (min: {minX}, {minY:.2f} mm)')

        # Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Statistical Shape Relation', fontdict={'fontsize': 16, 'fontweight': 'bold'})
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial')
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial')

        # Mark and annotate minimum points
        plt.scatter(minX, minY, color=line2.get_color(), zorder=3)

        # Add legend in top-left
        plt.legend(loc='upper left', fontsize=12, frameon=False)

        plt.show()
    def variousReconstructionStrategies():
        """
        Analyzes and compares performance across multiple pelvis reconstruction strategies.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures by 
        loading validation error data from different reconstruction methodologies and generating 
        comparative bar chart visualizations to assess the effectiveness of each approach.
        
        Key Steps:
        1. Load validation error data from four different reconstruction strategies
        2. Aggregate mean validation errors across all cross-validation folds (0-9)
        3. Process errors for optimal component configurations for each strategy
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate comparative bar chart showing mean errors with standard deviations
        6. Annotate bars with precise error values for quantitative comparison
        
        Reconstruction Strategies Compared:
        - Affine Transform: Uses rigid SVD + affine CPD transformations
        - Radial Basis Function: Adds RBF interpolation for enhanced non-rigid deformation
        - Statistical Shape Optimization: PCA-based shape models with least squares optimization
        - Statistical Shape Relation: PCA + regression mapping between feature and shape spaces
        
        Methodology Analysis:
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Optimal Components: Uses predetermined optimal number of components for fair comparison
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Bar chart showing mean validation errors with error bars (standard deviations)
        - Precise error value annotations on each bar for quantitative assessment
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating comprehensive strategy comparison
        
        Statistical Summary:
        - Computes mean and standard deviation validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual ranking of reconstruction strategies by performance
        - Enables identification of the most effective reconstruction approach
        
        Note: This comprehensive comparison helps determine the optimal reconstruction strategy
        by evaluating performance across multiple validation folds using consistent evaluation metrics.
        """
        # Initializing
        print("\t Initializing ...");
        affineTransformValidFolder = crossValidationFolder + "/AffineDeformation/BoneMuscleStructures";
        rbfBasedValidFolder = crossValidationFolder + "/RadialBasicFunctionStrategy/BoneMuscleStructures";
        shapeOptimBasedValidFolder = crossValidationFolder + "/ShapeOptimizationStrategy/BoneMuscleStructures";
        shapeRelationBasedValidFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;
        shapeOptimBasedOptimNumComps = 21; shapeRelationBasedOptimComps = 38;
    
        # Reading errors
        print("\t Reading errors ...");
        print("\t\t For affine base ...");
        affineTransformBasedValidErrors = [];
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(affineTransformValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            affineTransformBasedValidErrors.append(meanValidError);
        affineTransformBasedValidErrors = np.array(affineTransformBasedValidErrors);

        print("\t\t For radial basis function based ...");
        radialBasisFunctionBasedValidErrors = [];
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            validErrors = sp.readMatrixFromCSVFile(rbfBasedValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            meanValidError = np.mean(validErrors);
            radialBasisFunctionBasedValidErrors.append(meanValidError);
        radialBasisFunctionBasedValidErrors = np.array(radialBasisFunctionBasedValidErrors);

        print("\t\t For shape optimization based ...");
        shapeOptimizationBasedValidErrors = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(shapeOptimBasedValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                shapeOptimizationBasedValidErrors[validIndex,  numComps - 1] = validError;
        shapeOptimizationBasedValidErrors = np.array(shapeOptimizationBasedValidErrors);
        optimalValidErrors_shapeOptim = shapeOptimizationBasedValidErrors[:, shapeOptimBasedOptimNumComps - 1];
    
        print("\t\t For shape relation based ...");
        shapeRelationBasedValidErrors = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(shapeRelationBasedValidFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                shapeRelationBasedValidErrors[validIndex,  numComps - 1] = validError;
        shapeRelationBasedValidErrors = np.array(shapeRelationBasedValidErrors);
        optimalValidErrors_shapeRelation = shapeRelationBasedValidErrors[:, shapeRelationBasedOptimComps - 1];
    
        print("\t Drawing charts ...");
        ## Organize data
        groups = [
            ("Affine Transform", affineTransformBasedValidErrors*1000),
            ("Radial Basic \nFunction", radialBasisFunctionBasedValidErrors*1000),
            ("Statistical Shape \nOptimization", optimalValidErrors_shapeOptim*1000),
            ("Statistical Shape \nRelation", optimalValidErrors_shapeRelation*1000)
        ]
        ## Computing means and stds
        means = [[np.mean(data)] for _, data in groups]
        stds = [[np.std(data)] for _, data in groups]
        ## Plotting
        labels = [g[0] for g in groups]
        x = np.arange(len(labels))  # label locations
        width = 0.35  # width of the bars
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, [m[0] for m in means], width, yerr=[s[0] for s in stds], capsize=5)
        ## Add text labels for the means and decoration
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points", ha='center', va='bottom', fontsize=14, fontweight='bold')
        add_labels(bars, [m[0] for m in means])
        ax.set_ylabel('Validation Errors (mm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature-to-Structure Reconstruction Strategies', fontsize=14, fontweight='bold')
        ax.set_title('Comparison of Error Metrics Across Methods', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha='center', fontsize=12, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        boldFont = FontProperties(weight='bold', size=14)
        ax.legend(prop=boldFont)
        plt.tight_layout()
        plt.show()
    def boneMuscleStructureWithMultiLinearRegression():
        """
        Analyzes and visualizes the performance of multivariate linear regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        multivariate linear regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from multivariate linear regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + multivariate linear regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: This analysis focuses on multivariate linear regression as the mapping function
        between PCA-reduced feature and vertex parameter spaces for pelvis reconstruction.
        """
        # Initializing
        print("Initializing ...");
        multiLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_multiLinear = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(multiLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_multiLinear[validIndex,  numComps - 1] = validError;
        validationErrors_multiLinear = np.array(validationErrors_multiLinear);

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_multiLinear, axis=0) * 1000;
        N = validationErrors_multiLinear.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Multivariate Linear Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='center');
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithRidgeLinearRegression():
        """
        Analyzes and visualizes the performance of ridge linear regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        ridge linear regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from ridge linear regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + ridge linear regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Ridge regression adds L2 regularization to prevent overfitting compared to standard linear regression.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        ridgeLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_RidgeLinearRegression";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_ridgeLinearRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeLinearRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeLinearRegression = np.array(validationErrors_ridgeLinearRegression);

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_ridgeLinearRegression, axis=0) * 1000;
        N = validationErrors_ridgeLinearRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Ridge Linear Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='center');
        plt.show();
    def boneMuscleStructureWithCanonicalCorrelationAnalysisRegression():
        """
        Analyzes and visualizes the performance of canonical correlation analysis regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        canonical correlation analysis (CCA) regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from canonical correlation analysis bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + canonical correlation analysis mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Canonical correlation analysis finds linear combinations that maximize correlation between 
        feature and vertex parameter spaces, providing enhanced modeling of inter-space relationships.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        canonicalCorrelationRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_CanonicalCorrelationAnalysis";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;
        
        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_canonicalCorrelationRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(canonicalCorrelationRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_canonicalCorrelationRegression[validIndex,  numComps - 1] = validError;
        validationErrors_canonicalCorrelationRegression = np.array(validationErrors_canonicalCorrelationRegression);
        
        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_canonicalCorrelationRegression, axis=0) * 1000;
        N = validationErrors_canonicalCorrelationRegression.shape[1];
        x = np.arange(1, N + 1); y = drawingData;
        ## Find minimum points
        min_index = np.argmin(y);
        min_x = x[min_index]; min_y = y[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Canonical Correlation Analysis', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithPartialLeastSquaresRegression():
        """
        Analyzes and visualizes the performance of partial least squares regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        partial least squares (PLS) regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from partial least squares regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + partial least squares regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Partial least squares regression maximizes covariance between feature and vertex parameter spaces.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """

        # Initializing
        print("Initializing ...");
        partialLeastSquaresRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_PartialLeastSquaresRegression";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_partialLeastSquaresRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(partialLeastSquaresRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_partialLeastSquaresRegression[validIndex,  numComps - 1] = validError;
        validationErrors_partialLeastSquaresRegression = np.array(validationErrors_partialLeastSquaresRegression);
        optimalValidErrors_partialLeastSquaresRegression = validationErrors_partialLeastSquaresRegression[:, optimNumCompBoneMuscleOptim - 1];

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_partialLeastSquaresRegression, axis=0) * 1000;
        N = validationErrors_partialLeastSquaresRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Partial Least Squares Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithGaussianProcessRegression():
        """
        Analyzes and visualizes the performance of Gaussian process regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        Gaussian process regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from Gaussian process regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + Gaussian process regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Gaussian process regression provides probabilistic predictions and can capture 
        non-linear relationships between feature and vertex parameter spaces with uncertainty quantification.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """

        # Initializing
        print("Initializing ...");
        gaussianProcessRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_GaussianProcessRegressor";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_gaussianProcessRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(gaussianProcessRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_gaussianProcessRegression[validIndex,  numComps - 1] = validError;
        validationErrors_gaussianProcessRegression = np.array(validationErrors_gaussianProcessRegression);
        optimalValidErrors_gaussianProcessRegression = validationErrors_gaussianProcessRegression[:, optimNumCompBoneMuscleOptim - 1];

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_gaussianProcessRegression, axis=0) * 1000;
        N = validationErrors_gaussianProcessRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Gaussian Process Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithRidgeMultiOutputRegression():
        """
        Analyzes and visualizes the performance of ridge multi-output regression on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures using 
        ridge multi-output regression by loading validation error data across varying numbers 
        of principal components and generating performance visualizations with trend analysis.
        
        Key Steps:
        1. Load validation error data from ridge multi-output regression bone-muscle structure results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for varying numbers of components (1-80)
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate line plot visualization showing component count vs validation errors
        6. Identify and annotate optimal number of components with minimum error
        
        Methodology Analysis:
        - Shape Relation Strategy: Uses PCA + ridge multi-output regression mapping between feature and shape parameter spaces
        - Validation Framework: Processes errors across multiple cross-validation folds
        - Component Analysis: Evaluates performance across 1-80 principal components
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Line plot showing validation errors vs number of components
        - Minimum error point identification and annotation with coordinates
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating the specific regression method being analyzed
        
        Statistical Summary:
        - Computes mean validation errors across all folds for each component count
        - Converts measurements from meters to millimeters for clinical relevance
        - Identifies optimal number of components for minimum reconstruction error
        - Provides visual assessment of method performance and parameter sensitivity
        
        Note: Ridge multi-output regression combines ridge regularization with multi-output capability
        to handle multiple target variables simultaneously while preventing overfitting.
        Accuracy is computed only on bone vertices since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        ridgeMultiOutputRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_MultiOutputRegressor";
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;
        
        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_ridgeMultiOutputRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeMultiOutputRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeMultiOutputRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeMultiOutputRegression = np.array(validationErrors_ridgeMultiOutputRegression);
        optimalValidErrors_ridgeMultiOutputRegression = validationErrors_ridgeMultiOutputRegression[:, optimNumCompBoneMuscleOptim - 1];

        # Draw graphs
        print("Drawing the graphs ...");
        ## Forming drawing data
        drawingData = np.mean(validationErrors_ridgeMultiOutputRegression, axis=0) * 1000;
        N = validationErrors_ridgeMultiOutputRegression.shape[1];
        x = np.arange(1, N + 1);
        y1 = drawingData;
        ## Find minimum points
        min_index = np.argmin(y1);
        min_x = x[min_index];
        min_y = y1[min_index];
        ## Plotting the curves
        line1, = plt.plot(x, y1, label=f'Bone Structure (min: {min_x}, {min_y:.2f} mm)');
        ## Axis labels and title
        plt.xlabel('Numbers of Components', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.ylabel('Validation Errors (mm)', fontdict={'fontsize': 14, 'fontweight': 'bold'});
        plt.title('Ridge Multiple Ouput Regression', fontdict={'fontsize': 16, 'fontweight': 'bold'});
        plt.xticks(fontsize=12, fontweight='bold', fontname='Arial');
        plt.yticks(fontsize=12, fontweight='bold', fontname='Arial');
        ## Mark and annotate minimum points
        plt.scatter(min_x, min_y, color=line1.get_color(), zorder=3);
        plt.annotate(f'({min_x}, {min_y:.2f})', xy=(min_x, min_y), xytext=(min_x + 0.05, min_y + 0.05),  fontsize=14, color='red', ha='right');
        print("minx = ", min_x, "miny = ", min_y);
        plt.show();

        # Finished
        print("\t Finished.");
    def boneMuscleStructureWithWithVariousRegressions():
        """
        Analyzes and compares the performance of various regression models on bone-muscle structures.
        
        This function evaluates the reconstruction accuracy of pelvis bone-muscle structures by 
        loading validation error data from multiple regression approaches and generating comparative 
        bar chart visualizations to assess the effectiveness of each regression method.
        
        Key Steps:
        1. Load validation error data from six different regression model results
        2. Aggregate validation errors across all cross-validation folds (0-9)
        3. Process errors for optimal component configurations for each regression method
        4. Convert error measurements to millimeters for clinical presentation
        5. Generate comparative bar chart showing mean errors with standard deviations
        6. Annotate bars with precise error values for quantitative comparison
        
        Regression Models Compared:
        - Canonical Correlation Analysis: Finds maximally correlated linear combinations between spaces
        - Gaussian Process Regression: Non-parametric Bayesian approach with uncertainty quantification
        - Multivariate Linear Regression: Direct linear mapping between feature and vertex parameters
        - Partial Least Squares Regression: Maximizes covariance between feature and vertex spaces
        - Ridge Linear Regression: Linear regression with L2 regularization to prevent overfitting
        - Multi-output Ridge Regression: Ridge regression with multi-output capability
        
        Methodology Analysis:
        - Validation Framework: Processes errors across multiple cross-validation folds (0-9)
        - Optimal Components: Uses predetermined optimal number of components for each method
        - Error Metrics: Point-to-point distance measurements between predicted and ground truth
        - Feature Selection: Uses feature selection strategy index 10 for consistent evaluation
        
        Visualization Output:
        - Bar chart showing mean validation errors with error bars (standard deviations)
        - Precise error value annotations on each bar for quantitative assessment
        - Professional formatting with bold fonts and clear axis labeling
        - Title indicating comprehensive regression method comparison
        
        Statistical Summary:
        - Computes mean and standard deviation validation errors across all folds
        - Converts measurements from meters to millimeters for clinical relevance
        - Provides visual ranking of regression methods by performance
        - Enables identification of the most effective regression approach
        
        Note: This comprehensive comparison helps determine the optimal regression methodology
        for feature-to-pelvis reconstruction by evaluating performance across multiple validation 
        folds using consistent evaluation metrics. Accuracy is computed only on bone vertices 
        since muscle ground truth may not be available.
        """
        # Initializing
        print("Initializing ...");
        multiLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures";
        ridgeLinearRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_RidgeLinearRegression";
        canonicalCorrelationRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_CanonicalCorrelationAnalysis";
        partialLeastSquaresRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_PartialLeastSquaresRegression";
        gaussianProcessRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_GaussianProcessRegressor";
        ridgeMultiOutputRegresionFolder = crossValidationFolder + "/ShapeRelationStrategy/BoneMuscleStructures_MultiOutputRegressor";
        optimNumComps_multiLinerRegression = 38;
        optimNumComps_ridgeLinearRegression = 79;
        optimNumComps_canonicalRegression = 12;
        optimNumComps_partialLeastSquaresRegression = 38;
        optimNumComps_gaussianProcessRegression = 38;
        optimNumComps_ridgeMultiOutputRegression = 79;
        maxNumComps = 80; minNumComps = 1; minValidIndex = 0; maxValidIndex = 9;

        # Reading validating errors 
        print("Reading valiating errors ...");
        validationErrors_multiLinear = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(multiLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_multiLinear[validIndex,  numComps - 1] = validError;
        validationErrors_multiLinear = np.array(validationErrors_multiLinear);
        optimalValidErrors_multiLinear = validationErrors_multiLinear[:, optimNumComps_multiLinerRegression - 1];

        validationErrors_ridgeLinearRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeLinearRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeLinearRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeLinearRegression = np.array(validationErrors_ridgeLinearRegression);
        optimalValidErrors_ridgeLinearRegression = validationErrors_ridgeLinearRegression[:, optimNumComps_ridgeLinearRegression - 1];
    
        validationErrors_canonicalCorrelationRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(canonicalCorrelationRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_canonicalCorrelationRegression[validIndex,  numComps - 1] = validError;
        validationErrors_canonicalCorrelationRegression = np.array(validationErrors_canonicalCorrelationRegression);
        optimalValidErrors_canonicalCorrelationRegression = validationErrors_canonicalCorrelationRegression[:, optimNumComps_canonicalRegression - 1];
    
        validationErrors_partialLeastSquaresRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(partialLeastSquaresRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_partialLeastSquaresRegression[validIndex,  numComps - 1] = validError;
        validationErrors_partialLeastSquaresRegression = np.array(validationErrors_partialLeastSquaresRegression);
        optimalValidErrors_partialLeastSquaresRegression = validationErrors_partialLeastSquaresRegression[:, optimNumComps_partialLeastSquaresRegression - 1];
    
        validationErrors_gaussianProcessRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(gaussianProcessRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_gaussianProcessRegression[validIndex,  numComps - 1] = validError;
        validationErrors_gaussianProcessRegression = np.array(validationErrors_gaussianProcessRegression);
        optimalValidErrors_gaussianProcessRegression = validationErrors_gaussianProcessRegression[:, optimNumComps_gaussianProcessRegression - 1];

        validationErrors_ridgeMultiOutputRegression = np.zeros((maxValidIndex + 1, maxNumComps));
        for validIndex in range(minValidIndex, maxValidIndex + 1):
            for numComps in range(minNumComps, maxNumComps + 1):
                validErrors = sp.readMatrixFromCSVFile(ridgeMultiOutputRegresionFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv");
                validError = validErrors.mean();
                validationErrors_ridgeMultiOutputRegression[validIndex,  numComps - 1] = validError;
        validationErrors_ridgeMultiOutputRegression = np.array(validationErrors_ridgeMultiOutputRegression);
        optimalValidErrors_ridgeMultiOutputRegression = validationErrors_ridgeMultiOutputRegression[:, optimNumComps_ridgeMultiOutputRegression - 1];
    
        print("\t Drawing charts ...");

        ## Organize data
        groups = [
            ("Canonical\nCorrelation", optimalValidErrors_canonicalCorrelationRegression * 1000),
            ("Gaussian\nProcess", optimalValidErrors_gaussianProcessRegression * 1000),
            ("Multivariate\nLinear", optimalValidErrors_multiLinear * 1000),
            ("Partial\nLeastSquare", optimalValidErrors_partialLeastSquaresRegression * 1000),
            ("Ridge\nLinear", optimalValidErrors_ridgeLinearRegression * 1000),
            ("Multioutput\nRidge", optimalValidErrors_ridgeMultiOutputRegression * 1000)
        ]

        ## Compute means and standard deviations
        means = [np.mean(data) for _, data in groups]
        stds = [np.std(data) for _, data in groups]
        labels = [name for name, _ in groups]

        ## Set x locations and bar width
        x = np.arange(len(labels))  # Centered positions
        width = 0.6

        ## Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, width=width, yerr=stds, capsize=5)

        ## Add value labels on top of each bar
        for bar, val in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=13, fontweight='bold')

        ## Axis labels and ticks
        ax.set_ylabel('Validation Errors (mm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Multiple Regression Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center')
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold')

        ## No legend needed (you have no groups/colors)
        ax.grid(False)

        ## Layout
        plt.tight_layout()
        plt.style.use('default');
        plt.show()
    
    # Running
    print("Running ...");
    boneMuscleStructureWithWithVariousRegressions();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_trainValidationVariousFeatures():
    """
    Performs comprehensive training and validation of multi-output ridge regression for pelvis bone-muscle reconstruction across various feature selection strategies.
    
    This function evaluates the accuracy of pelvis bone reconstruction by training multi-output ridge regression models 
    on different feature selection strategies and varying numbers of principal components. The evaluation is performed 
    across multiple cross-validation folds to ensure robust performance assessment.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds, components)
    2. Initialize file paths and load template data from personalized pelvis structures
    3. Load feature selection strategies and template mesh information
    4. For each feature selection strategy and validation fold:
       - Load training and validation subject IDs from cross-validation splits
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction to both feature and vertex spaces
       - Train multi-output ridge regression model with cross-validation
       - For each number of components:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Compute point-to-point distance errors (bone vertices only)
       - Save validation errors to CSV files
    
    Command Line Arguments:
    - StartFeatureIndex: Starting index for feature selection strategies
    - EndFeatureIndex: Ending index for feature selection strategies  
    - StartValidationIndex: Starting validation fold index
    - EndValidationIndex: Ending validation fold index
    - StartNumComponents: Starting number of PCA components
    - EndNumComponents: Ending number of PCA components
    
    Data Processing:
    - Uses personalized pelvis bone-muscle meshes as ground truth
    - Applies barycentric landmark reconstruction for feature point extraction
    - Implements standardized scaling for numerical stability
    - Uses PCA for dimensionality reduction in both input and output spaces
    
    Model Training:
    - Multi-output ridge regression with cross-validation (RidgeCV)
    - Automatic hyperparameter tuning through cross-validation
    - Handles multiple target variables simultaneously
    - Prevents overfitting through L2 regularization
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    - Validation errors organized by feature strategy, validation fold, and component count
    - Performance metrics for systematic evaluation of reconstruction accuracy
    
    Note: This function focuses on bone vertex accuracy since muscle ground truth may not be available.
    The comprehensive evaluation across multiple parameters enables optimal model selection and performance assessment.
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 7):
        print("Please input the command as the following: [ProgramName] [StartFeatureIndex] [EndFeatureIndex] [StartValidationIndex] [EndValidationIndex] [StartNumComponents] [EndNumComponents]");
        return;
    startFeatureIndex = int(sys.argv[1]);
    endFeatureIndex = int(sys.argv[2]);
    startValidationIndex = int(sys.argv[3]);
    endValidationIndex = int(sys.argv[4]);
    startNumComponents = int(sys.argv[5]);
    endNumComponents = int(sys.argv[6]);
    disk = "I:";
    mainFolder = disk + r"/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + r"/Data/PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + r"/FemalePelvisGeometries";
    personalizedFemalePelvisFolder = femalePelvisFolder + r"/PersonalizedPelvisStructures";
    crossValidationFolder = mainFolder + r"/Data/PelvisBoneRecon/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    trainTestSplitFolder = crossValidationFolder + "/TrainTestSplits";
    validationErrorFolder = crossValidationFolder + "/ValidationErrors";
    testingErrorFolder = crossValidationFolder + "/TestingErrors";
    featureSelectionProtocolFolder = mainFolder + r"/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    templateFolder = mainFolder + r"/Data/Template";

    # Reading structural feature strategies
    print("Reading structural feature strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    for featureSelIndex in range(startFeatureIndex, endFeatureIndex + 1):
        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(templatePelvisBoneMuscleMesh, tempPelvicFeatures);

        # Iterate for each validation index
        for validIndex in range(startValidationIndex, endValidationIndex + 1):
            # Debugging information
            print("/********************************************** VALIDATION: ", validIndex);

            # Reading training and validating subject IDs
            print("\t Reading training and validating subject IDs ...");
            trainIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TrainingIDs_{validIndex}.txt");
            validIDs = sp.readListOfStrings(trainTestSplitFolder + f"/ValidationIDs_{validIndex}.txt");

            # Forming the training and validating data
            print("\t Forming the training and validating data ...");
            trainingPelvicVertexData = []; validPelvicVertexData = [];
            trainingPelvicFeatureData = []; validPelvicFeatureData = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(validIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                validPelvicVertexData.append(pelvicStructure.vertices.flatten());
                validPelvicFeatureData.append(pelvicFeatures.flatten());
            
            # Iterate for each number of components
            for numComps in range(startNumComponents, endNumComponents + 1):
                print("/***************************** FEATURE SELECTION INDEX: ", featureSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", numComps);

                # Scale the pelvic feature and vertex data
                print("\t\t Scaling the pelvic feature and vertex data ...");
                yScaler = StandardScaler().fit(trainingPelvicVertexData);
                xScaler = StandardScaler().fit(trainingPelvicFeatureData);
                scaledYData = yScaler.transform(trainingPelvicVertexData);
                scaledXData = xScaler.transform(trainingPelvicFeatureData);

                # Control the number of components
                print("\t\t Controlling the number of components ...");
                targetNumComps = numComps;
                xDims = scaledXData.shape[1];
                yDims = scaledYData.shape[1];
                xNumComps = min(xDims, targetNumComps);
                yNumComps = min(yDims, targetNumComps);

                # Parameterize the scaled pelvic vertex and feature data
                print("\t\t Parameterizing the scaled pelvic vertex and feature data ...");
                xSSM = PCA(n_components=xNumComps);
                xSSM.fit(scaledXData);
                trainXParamData = xSSM.transform(scaledXData);
                ySSM = PCA(n_components=yNumComps);
                ySSM.fit(scaledYData);
                trainYParamData = ySSM.transform(scaledYData);

                # Linear regression from feature parameters to vertex parameters
                print("\t\t Linear regression from feature parameters to vertex parameters ...");
                model = MultiOutputRegressor(RidgeCV())
                model.fit(trainXParamData, trainYParamData)

                # Computing errors on the validating data
                print("\t\t Computing errors on the validating data ...");
                ## Scale and parameterize the validation data
                scaledValidXData = xScaler.transform(validPelvicFeatureData);
                validXParams = xSSM.transform(scaledValidXData);
                ## Try to predict the pelvic vertex params from the pelvic feature params
                predYParams = model.predict(validXParams);
                ## Inverse transform to the scaled Y data
                predYParams = predYParams.reshape(-1, yNumComps);
                predScaledYData = ySSM.inverse_transform(predYParams);
                predYData = yScaler.inverse_transform(predScaledYData);
                ## Computing validating errors
                avgP2PDists = [];
                for v, predY in enumerate(predYData):
                    # Getting the valiation data and predicted data
                    validY = validPelvicVertexData[v];
                    validPelvicStructureVertices = validY.reshape(-1, 3);
                    predPelvicStructureVertices = predY.reshape(-1, 3);

                    # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                    validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices];
                    predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                    # Compute points to points distances
                    avgP2PDist = sp.computeAveragePointsToPointsDistance(validPelvisBoneVertices, predPelvisBoneVertices);
                    avgP2PDists.append(avgP2PDist);
                avgP2PDists = np.array(avgP2PDists);

                # Save the computed errors
                print("\t\t Saving the computed errors ...", end=' ', flush=True);
                print("-> Grand Mean: ", avgP2PDists.mean());
                sp.saveMatrixToCSVFile(validationErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.csv", avgP2PDists);
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_determineOptimalNumComponents():
    """
    Determines the optimal number of principal components for multi-output ridge regression across feature selection strategies.
    
    This function analyzes validation error data from pelvis bone-muscle reconstruction experiments to identify 
    the optimal number of PCA components that minimizes reconstruction errors for each feature selection strategy.
    The analysis is performed across multiple cross-validation folds to ensure robust optimization.
    
    Key Steps:
    1. Initialize file paths and load feature selection strategy configurations
    2. For each feature selection strategy (0 to max available strategies):
       - Load validation error data across all cross-validation folds (0-9)
       - Process errors for varying numbers of components (1-200)
       - Aggregate validation errors from CSV files for each fold and component count
       - Compute mean validation errors across all folds for each component number
       - Identify the optimal number of components that minimizes mean validation error
       - Store optimal component count for the current strategy
    3. Save all optimal component counts to CSV file for future use
    
    Data Processing:
    - Reads validation errors from CSV files organized by feature strategy, fold, and component count
    - Processes errors across extended component range (1-200) due to data augmentation
    - Computes statistical aggregation across 10-fold cross-validation
    - Identifies minimum error points for parameter optimization
    
    Optimization Strategy:
    - Grid search approach across component numbers to find global minimum
    - Cross-validation averaging to ensure robust parameter selection
    - Feature strategy-specific optimization for tailored performance
    
    Output:
    - CSV file containing optimal number of components for each feature selection strategy
    - Enables informed parameter selection for final model training and testing
    - Provides foundation for subsequent performance evaluation with optimal settings
    
    Note: This function serves as a critical preprocessing step for model optimization,
    ensuring that each feature selection strategy uses its optimal number of PCA components
    for maximum reconstruction accuracy in the final evaluation phase.
    """

    # Initializing
    print("Initializing ...");
    disk = "I:";
    mainFolder = disk + r"/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + r"/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisReconFolder + r"/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    featureSelectionProtocolFolder = mainFolder + r"/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    
    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    optimNumCompsBuffer = [];
    for featureSelIndex in range(len(featureSelectionStrategyDict)):
        # Debugging information
        print("/********************************************** FEATURE SELECTION INDEX: ", featureSelIndex);

        # Reading the validation errors for ten fold with 1 to 200 num of components
        print("\t Reading the validation errors for ten fold with 1 to 200 num of components ...");
        validationErrors = np.zeros((10, 200));
        for validIndex in range(10):
            for numComps in range(1, 201):
                # Reading the validation errors
                print("\t\t Validating index: ", validIndex, ", numComps: ", numComps);
                validErrors = sp.loadNumPYArrayFromNPY(crossValidationFolder + 
                                                       f"/ValidationErrors/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}_{numComps}.npy");
                validError = validErrors.mean();
                validationErrors[validIndex, numComps - 1] = validError;

        # Compute the mean errors for all folds of cross-validation
        print("\t Computing the mean errors for all folds of cross-validation ...");
        meanValidationErrors = np.mean(validationErrors, axis=0);

        # Estimate the optimal number of components
        print("\t Estimating the optimal number of components: ", end="", flush=True);
        optimNumComps = np.argmin(meanValidationErrors) + 1;  # +1 because numComps starts from 1
        print(optimNumComps);

        # Add optim numComps to buffer
        optimNumCompsBuffer.append(optimNumComps);

    # Save the optimal number of components
    print("Saving the optimal number of components ...");
    optimNumCompsBuffer = np.array(optimNumCompsBuffer);
    sp.saveMatrixToCSVFile(crossValidationFolder + "/OptimalNumComponents.csv", optimNumCompsBuffer);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_testUsingOptimalNumComponents():
    """
    Performs comprehensive testing of multi-output ridge regression for pelvis bone-muscle reconstruction using optimal component configurations.
    
    This function evaluates the accuracy of pelvis bone reconstruction by testing pre-trained multi-output ridge regression models 
    on unseen test data using optimal numbers of principal components determined from validation experiments. The evaluation is 
    performed across multiple feature selection strategies and cross-validation folds to provide robust performance assessment.
    
    Key Steps:
    1. Parse command line arguments for processing range (feature strategies, validation folds)
    2. Initialize file paths and load template data from personalized pelvis structures
    3. Load feature selection strategies, template mesh information, and optimal component counts
    4. For each feature selection strategy and validation fold:
       - Load training and testing subject IDs from cross-validation splits
       - Extract pelvis bone-muscle vertices and feature points for each subject
       - Scale and normalize both feature and vertex data using StandardScaler
       - Apply PCA dimensionality reduction using optimal number of components
       - Train multi-output ridge regression model with cross-validation on training data
       - For each testing subject:
         * Transform features to PCA parameter space
         * Predict vertex parameters using trained regression model
         * Reconstruct pelvis bone-muscle mesh from predicted parameters
         * Save predicted meshes for visualization and analysis
         * Compute point-to-point distance errors (bone vertices only)
       - Save testing errors to CSV files
    
    Command Line Arguments:
    - StartFeatureIndex: Starting index for feature selection strategies
    - EndFeatureIndex: Ending index for feature selection strategies  
    - StartValidationIndex: Starting validation fold index
    - EndValidationIndex: Ending validation fold index
    
    Data Processing:
    - Uses personalized pelvis bone-muscle meshes as ground truth
    - Applies barycentric landmark reconstruction for feature point extraction
    - Implements standardized scaling for numerical stability
    - Uses PCA for dimensionality reduction with pre-determined optimal components
    
    Model Training and Testing:
    - Multi-output ridge regression with cross-validation (RidgeCV)
    - Automatic hyperparameter tuning through cross-validation
    - Tests on completely unseen data separate from training/validation sets
    - Prevents overfitting through L2 regularization and proper data splits
    
    Output:
    - CSV files containing average point-to-point distances for each configuration
    - Testing errors organized by feature strategy and validation fold
    - Predicted pelvis bone and bone-muscle meshes saved for visualization
    - Performance metrics for final model evaluation on test data
    
    Note: This function performs final testing using optimal parameters determined from validation,
    providing unbiased performance estimates on completely unseen test data. Accuracy is computed 
    only on bone vertices since muscle ground truth may not be available.
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Please input the command as the following: [ProgramName] [StartFeatureIndex] [EndFeatureIndex] [StartValidationIndex] [EndValidationIndex]");
        return;
    startFeatureIndex = int(sys.argv[1]);
    endFeatureIndex = int(sys.argv[2]);
    startValidationIndex = int(sys.argv[3]);
    endValidationIndex = int(sys.argv[4]);
    disk = "I:";
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    femalePelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    personalizedFemalePelvisFolder = femalePelvisFolder + "/PersonalizedPelvisStructures";
    crossValidationFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    trainTestSplitFolder = crossValidationFolder + "/TrainTestSplits";
    testingErrorFolder = crossValidationFolder + "/TestingErrors";
    predictedTestingPelvicStructureFolder = crossValidationFolder + "/PredictedTestingPelvicStructures";    
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    templateFolder = mainFolder + "/Data/Template";

    # Reading structural feature strategies
    print("Reading structural feature strategies ...");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Read optimal number of components
    print("Reading optimal number of components ...");
    optimNumComps = sp.readMatrixFromCSVFile(crossValidationFolder + "/OptimalNumComponents.csv");
    optimNumComps = optimNumComps.flatten();

    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    for featureSelIndex in range(startFeatureIndex, endFeatureIndex + 1):
        # Get the feature selection strategies
        print("\t Getting the feature selection strategy ...");
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(templatePelvisBoneMuscleMesh, tempPelvicFeatures);
        targetNumComps = optimNumComps[featureSelIndex];

        # Iterate for each validation index
        for validIndex in range(startValidationIndex, endValidationIndex + 1):
            # Debugging information
            print("/***************************** FEATURE SELECTION INDEX: ", featureSelIndex, ", VALIDATION INDEX: ", validIndex, ", NUM COMPS: ", targetNumComps);

            # Reading training and testing subject IDs
            print("\t Reading training and testing subject IDs ...");
            trainIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TrainingIDs_{validIndex}.txt");
            testIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TestingIDs_{validIndex}.txt");

            # Forming the training and testing data
            print("\t Forming the training and testing data ...");
            trainingPelvicVertexData = []; testPelvicVertexData = [];
            trainingPelvicFeatureData = []; testPelvicFeatureData = [];
            testingPelvisMeshes = [];
            for i, ID in enumerate(trainIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                trainingPelvicVertexData.append(pelvicStructure.vertices.flatten());
                trainingPelvicFeatureData.append(pelvicFeatures.flatten());
            for i, ID in enumerate(testIDs):
                pelvicStructure = sp.readMesh(personalizedFemalePelvisFolder + f"/{ID}-PersonalizedPelvisBoneMuscleMesh.ply");
                pelvicFeatures = sp.reconstructLandmarksFromBarycentric(pelvicStructure, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords);
                testPelvicVertexData.append(pelvicStructure.vertices.flatten());
                testPelvicFeatureData.append(pelvicFeatures.flatten());
            
            # Scale the pelvic feature and vertex data
            print("\t\t Scaling the pelvic feature and vertex data ...");
            yScaler = StandardScaler().fit(trainingPelvicVertexData);
            xScaler = StandardScaler().fit(trainingPelvicFeatureData);
            scaledYData = yScaler.transform(trainingPelvicVertexData);
            scaledXData = xScaler.transform(trainingPelvicFeatureData);

            # Control the number of components
            print("\t\t Controlling the number of components ...");
            xDims = scaledXData.shape[1];
            yDims = scaledYData.shape[1];
            xNumComps = min(xDims, targetNumComps);
            yNumComps = min(yDims, targetNumComps);

            # Parameterize the scaled pelvic vertex and feature data
            print("\t\t Parameterizing the scaled pelvic vertex and feature data ...");
            xSSM = PCA(n_components=xNumComps);
            xSSM.fit(scaledXData);
            trainXParamData = xSSM.transform(scaledXData);
            ySSM = PCA(n_components=yNumComps);
            ySSM.fit(scaledYData);
            trainYParamData = ySSM.transform(scaledYData);

            # Linear regression from feature parameters to vertex parameters
            print("\t\t Linear regression from feature parameters to vertex parameters ...");
            model = MultiOutputRegressor(RidgeCV())
            model.fit(trainXParamData, trainYParamData)

            # Computing errors on the testing data
            print("\t\t Computing errors on the testing data ...");
            ## Scale and parameterize the testing data
            scaledTestXData = xScaler.transform(testPelvicFeatureData);
            testXParams = xSSM.transform(scaledTestXData);
            ## Try to predict the pelvic vertex params from the pelvic feature params
            predYParams = model.predict(testXParams);
            ## Inverse transform to the scaled Y data
            predYParams = predYParams.reshape(-1, yNumComps);
            predScaledYData = ySSM.inverse_transform(predYParams);
            predYData = yScaler.inverse_transform(predScaledYData);
            ## Computing testing errors
            avgP2PDists = [];
            for t, predY in enumerate(predYData):
                # Getting the testing data and predicted data
                testY = testPelvicVertexData[t];
                testPelvicStructureVertices = testY.reshape(-1, 3);
                predPelvicStructureVertices = predY.reshape(-1, 3);

                # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
                testPelvisBoneVertices = testPelvicStructureVertices[pelvisBoneVertexIndices];
                predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices];

                # Save the predicted pelvis bone muscle mesh
                testPelvisMesh = sp.formMesh(predPelvisBoneVertices, templatePelvisBoneMesh.faces);
                testPelvisBoneMuscleMesh = sp.formMesh(predPelvicStructureVertices, templatePelvisBoneMuscleMesh.faces);
                testPelvisFilePath = predictedTestingPelvicStructureFolder + f"/PredictedPelvicStructure_{featureSelIndex}_{validIndex}_{t}.ply";
                testPelvisMuscleFilePath = predictedTestingPelvicStructureFolder + f"/PredictedPelvicMuscle_{featureSelIndex}_{validIndex}_{t}.ply";
                sp.saveMeshToPLY(testPelvisFilePath, testPelvisMesh);
                sp.saveMeshToPLY(testPelvisMuscleFilePath, testPelvisBoneMuscleMesh);

                # Compute points to points distances
                avgP2PDist = sp.computeAveragePointsToPointsDistance(testPelvisBoneVertices, predPelvisBoneVertices);
                avgP2PDists.append(avgP2PDist);
            avgP2PDists = np.array(avgP2PDists);

            # Save the computed errors
            print("\t\t Saving the computed errors ...", end=' ', flush=True);
            print("-> Grand Mean: ", avgP2PDists.mean());
            sp.saveMatrixToCSVFile(testingErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv", avgP2PDists);
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrors():
    """
    Analyzes and visualizes testing errors from multi-output ridge regression pelvis reconstruction across feature selection strategies.
    
    This function evaluates the final testing performance of the optimal pelvis bone-muscle reconstruction model
    by loading testing error data across different feature selection strategies and generating comprehensive
    statistical visualizations to assess model accuracy and feature importance relationships.
    
    Key Steps:
    1. Initialize file paths and load feature selection strategy configurations
    2. Calculate the number of features for each feature selection strategy
    3. Load testing error data across all feature strategies and validation folds
    4. Aggregate testing errors and compute statistical measures (mean, standard deviation)
    5. Generate bar chart visualization with error bars and trend analysis
    6. Perform polynomial fitting to identify error trends with feature count
    7. Calculate theoretical zero-error point for feature requirement estimation
    8. Save visualization results for publication and analysis
    
    Data Processing:
    - Reads testing errors from CSV files organized by feature strategy and validation fold
    - Processes errors across all available feature selection strategies (0 to max)
    - Computes mean and standard deviation across 10-fold cross-validation
    - Converts measurements from meters to millimeters for clinical presentation
    
    Statistical Analysis:
    - Mean testing error computation across cross-validation folds
    - Standard deviation calculation for error bar representation
    - Linear polynomial fitting to identify feature count vs error relationship
    - Zero-error point estimation to determine theoretical feature requirements
    
    Visualization Output:
    - Bar chart showing mean testing errors with standard deviation error bars
    - Precise error value annotations on each bar for quantitative assessment
    - Polynomial trend line with equation and zero-error point estimation
    - Professional formatting with bold fonts and clear axis labeling
    - Legend showing regression equation and zero-error point coordinates
    
    Feature Analysis:
    - Maps feature selection strategy indices to actual feature counts
    - Analyzes relationship between number of anatomical features and reconstruction accuracy
    - Provides insights into optimal feature set size for clinical applications
    - Enables determination of minimum feature requirements for acceptable accuracy
    
    Output Files:
    - High-resolution PNG chart saved for publication use
    - Statistical summary data for performance reporting
    - Trend analysis results for feature optimization guidance
    
    Note: This function provides final performance evaluation on completely unseen test data,
    offering unbiased assessment of model generalization capability across different feature
    selection strategies. The polynomial fitting helps identify the theoretical minimum number
    of anatomical features required for zero reconstruction error.
    """

    # Initializing
    print("Initializing ...");
    disk = "I:";
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisReconFolder + "/CrossValidation/ShapeRelationStrategy/OptimalTrainValidTest";
    testingErrorFolder = crossValidationFolder + "/TestingErrors";
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    optimNumComps = sp.readMatrixFromCSVFile(crossValidationFolder + "/OptimalNumComponents.csv");

    # Estimate the number of features in each feature selection strategy
    print("Estimating the number of features in each feature selection strategy ...");
    numOfStrategyFeatures = [];
    for featureSelIndex in range(len(featureSelectionStrategyDict)):
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        numOfStrategyFeatures.append(len(featureSelectionIndices));
    numOfStrategyFeatures = np.array(numOfStrategyFeatures);

    # Reading testing errors
    print("Reading testing errors ...");
    drawingData = np.zeros((10, len(featureSelectionStrategyDict)));
    for featureSelIndex in range(len(featureSelectionStrategyDict)):
        for validIndex in range(10):
            # Reading the testing errors
            print(f"({featureSelIndex}, {validIndex})", end=' ', flush=True);
            testingErrors = sp.readMatrixFromCSVFile(testingErrorFolder + f"/AveragePoint2PointDistances_{featureSelIndex}_{validIndex}.csv");
            testingError = testingErrors.mean();
            drawingData[validIndex, featureSelIndex] = testingError;
    drawingData = np.array(drawingData); print("");

    # Draw bar charts
    print("Drawing bar charts ...");
    ## Forming drawing data
    means = np.mean(drawingData, axis=0) * 1000;
    stds = np.std(drawingData, axis=0) * 1000;
    ## Define label as the number of features in each feature selection strategy
    labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(len(featureSelectionStrategyDict))];
    ## Set x locations and bar width
    x_locs = np.arange(len(labels));
    bar_width = 0.75;
    ## Plotting the bar chart
    fig, ax = plt.subplots(figsize=(12, 6));
    bars = ax.bar(x_locs, means, width=bar_width, yerr=stds, capsize=5);
    ## Add value labels on top of each bar
    for bar, val in zip(bars, means):
        height = bar.get_height();
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10); # not bold to avoid overlap with error bars
    ## Draw the poly fit line using linear regression line
    poly_fit = np.polyfit(x_locs, means, 1);
    poly_fit_line = np.polyval(poly_fit, x_locs);
    ## Draw the poly fit line
    line = ax.plot(x_locs, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
    ## Find the zero error point by fitting another line with real number of feature points
    realPolyFit = np.polyfit(numOfStrategyFeatures[x_locs], means, 1);    
    ## Compute the zero error point when the mean error reach zero
    zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
    zero_error_y = np.polyval(realPolyFit, zero_error_x);
    ## Show the zero error point in the legend
    ax.legend([bars, line],
          [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
          loc='upper right', fontsize=12);
    ## Axis labels and ticks
    ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
    ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
    ax.set_xticks(x_locs);
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
    ax.set_yticks(ax.get_yticks());
    ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
    ## No legend needed (you have no groups/colors)
    ax.grid(False);
    ## Add a title
    ax.set_title('Testing Errors vs Number of Features in Feature Selection Strategies', fontsize=16, fontweight='bold');
    ## Layout
    plt.tight_layout();
    plt.show();
    ## Save the figure
    fig.savefig(crossValidationFolder + "/TestingErrorsBarChart.png", dpi=400, bbox_inches='tight');

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_computeVariousTestingErrors():
    """
    Computes comprehensive testing error metrics for multi-output ridge regression pelvis reconstruction across feature selection strategies.
    
    This function evaluates the final testing performance of the optimal pelvis bone-muscle reconstruction model
    by loading predicted and ground truth data to compute various error metrics including mesh accuracy, feature 
    accuracy, muscle attachment errors, and vertex-to-vertex distance measurements for comprehensive evaluation.
    
    Key Steps:
    1. Parse command line arguments for feature selection strategy range
    2. Initialize file paths and load template data for pelvis bone-muscle structures
    3. Load feature selection strategies and compute barycentric coordinates for feature extraction
    4. For each feature selection strategy and validation fold:
       - Load testing subject IDs from cross-validation splits
       - For each testing subject:
         * Load predicted pelvis bone-muscle mesh from reconstruction results
         * Load ground truth personalized pelvis structures and CT bone meshes
         * Perform rigid registration between predicted and ground truth meshes
         * Compute various error metrics:
           - Vertex-to-vertex distances on bone-muscle structures
           - Mesh-to-mesh distances for bone structures
           - Feature point reconstruction accuracy
           - Muscle attachment point accuracy
       - Aggregate error measurements across all testing subjects
    5. Save computed error metrics to files for analysis and visualization
    
    Command Line Arguments:
    - StartFeatureIndex: Starting index for feature selection strategies to process
    - EndFeatureIndex: Ending index for feature selection strategies to process
    
    Error Metrics Computed:
    - Mesh Testing Errors: Average point-to-point distances between predicted and CT bone meshes
    - Feature Testing Errors: Accuracy of anatomical feature point reconstruction
    - Muscle Attachment Errors: Accuracy at critical muscle attachment locations
    - Vertex-to-Vertex Distances: Comprehensive vertex-level accuracy on personalized meshes
    - Vertex-to-Mesh Distances: Accuracy against CT-reconstructed bone surfaces
    - Vertex-to-BoneMuscle Distances: Overall bone-muscle structure accuracy
    
    Data Processing:
    - Uses predicted pelvis structures from optimal multi-output ridge regression models
    - Compares against both personalized ground truth and CT-reconstructed references
    - Applies rigid registration for fair comparison between coordinate systems
    - Processes muscle attachment points within 6mm radius for clinical relevance
    
    Output Files:
    - CSV files containing aggregated error metrics for each feature strategy
    - NumPy arrays storing detailed vertex-level distance measurements
    - Organized by feature selection strategy for systematic analysis
    
    Note: This comprehensive error analysis provides multiple perspectives on reconstruction
    accuracy, enabling detailed assessment of model performance across different anatomical
    regions and clinical applications. The function focuses on clinically relevant metrics
    including muscle attachment accuracy and overall structural fidelity.
    """

    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Please input the command as the following: [ProgramName] [StartFeatureIndex] [EndFeatureIndex]");
        return;
    startFeatureIndex = int(sys.argv[1]);
    endFeatureIndex = int(sys.argv[2]);
    disk = "I:";
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    # debugFolder = pelvisReconFolder + "/Debugs";
    templateFolder = mainFolder + "/Data/Template";
    femalePelvisFolder = pelvisReconFolder + "/FemalePelvisGeometries";
    crossValidationFolder = pelvisReconFolder + "/CrossValidation";
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    optimalTrainValidTestFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest";
    predTestingPelvisStructureFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/PredictedTestingPelvicStructures";
    meshFeatureMuscleErrorFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/MeshFeatureMuscleErrors";
    numOfValids = 10; numOfTests = 30; numOfStrategies = len(featureSelectionStrategyDict);
    def estimateMuscleAttachmentIndices(pelvisBoneMesh, attachmentPoints, maxDist=0.006):
        # Define buffer as flatten vector
        outIndices = np.array([]);
    
        # Iterate for each attachment point
        for point in attachmentPoints:
            # Estimate the vertex indices within 6 mm
            indices = sp.estimateNearestIndicesWithinRadius(point, pelvisBoneMesh.vertices, inRadius=maxDist);
            indices = np.array(indices);
    
            # If indices are found, append vertical indices to the buffer
            if len(indices) > 0: 
                outIndices = np.concatenate([outIndices, indices.flatten()])

        # Return the unique indices
        return np.unique(outIndices).astype(np.int32);

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    templatePelvisFeaturePoints = sp.read3DPointsFromPPFile(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh_picked_points.pp");
    templatePelvisMuscleAttachmentPoints = sp.read3DPointsFromPPFile(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh_muscleAttachmentPoints.pp");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);
    pelvisBoneFeatureBaryCoords, pelvisBoneFeatureBaryIndices = sp.computeBarycentricCoordinates(templatePelvisBoneMesh, templatePelvisFeaturePoints);
    # muscleAttachmentIndices = estimateMuscleAttachmentIndices(templatePelvisBoneMesh, templatePelvisMuscleAttachmentPoints, 0.006);

    # Reading template information
    print("Reading template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/PelvisBonesMuscles/TempPelvisBoneMesh.ply");
    numOfPelvisBoneVertices = templatePelvisBoneMesh.vertices.shape[0];

    # Iterate for each feature selection strategy
    print("Iterating for each feature selection strategy ...");
    for featureSelIndex in range(startFeatureIndex, endFeatureIndex + 1):
        # Generate buffer for saving testing errors
        print("Generate buffer for saving testing errors ...");
        # meshTestingErrors = np.zeros((numOfValids, numOfTests));
        # featureTestingErrors = np.zeros((numOfValids, numOfTests));
        # muscleAttachmentTestingErrors = np.zeros((numOfValids, numOfTests));

        ## Iterate for each num of validation
        print("Iterating for each num of validation ...");
        vertex2VertexDistanceBuffer = [];
        vertex2VertexMeshDistanceBuffer = [];
        vertex2VertexBoneMuscleDistanceBuffer = [];
        for validIndex in range(numOfValids):
            # Reading testing IDs
            print("\t Reading testing IDs ...");
            testingIDFilePath = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TrainTestSplits/TestingIDs_" + str(validIndex) + ".txt";
            testingIDs = sp.readListOfStrings(testingIDFilePath);

            ## Iterate for each num of testing
            print("\t Iterating for each num of testing ...");
            for testIndex in range(numOfTests):
                # Debugging
                print(f"\t\t Comuting for the case: ({featureSelIndex}, {validIndex}, {testIndex})");

                # Reading the predicted testing pelvis structure
                pdPelvisBoneMuscleMesh = sp.readMesh(predTestingPelvisStructureFolder + f"/PredictedPelvicMuscle_{featureSelIndex}_{validIndex}_{testIndex}.ply");
                pdPelvisBoneMesh = sp.formMesh(pdPelvisBoneMuscleMesh.vertices[pelvisBoneVertexIndices], templatePelvisBoneMesh.faces);

                # Reading ground truth information
                gtPersonalizedBoneMuscleMesh = sp.readMesh(femalePelvisFolder + f"/AllPelvicStructures/{testingIDs[testIndex]}-PersonalizedPelvisBoneMuscleMesh.ply");
                gtPersonalizedBoneVertices = gtPersonalizedBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
                gtPersonalizedBoneMesh = sp.formMesh(gtPersonalizedBoneVertices, templatePelvisBoneMesh.faces);
                gtBoneMesh = sp.readMesh(femalePelvisFolder + f"/AllPelvicStructures/{testingIDs[testIndex]}-PelvisBoneMesh.ply");
                # gtPelvisBoneFeatures = sp.reconstructLandmarksFromBarycentric(gtPersonalizedBoneMesh, pelvisBoneFeatureBaryIndices, pelvisBoneFeatureBaryCoords);

                # Compute predicted information
                svdTransform = sp.estimateRigidSVDTransform(pdPelvisBoneMuscleMesh.vertices, gtPersonalizedBoneMuscleMesh.vertices);
                pdPelvisBoneMuscleMesh = sp.transformMesh(pdPelvisBoneMuscleMesh, svdTransform);
                # pdPelvisBoneFeatures = sp.reconstructLandmarksFromBarycentric(pdPelvisBoneMesh, pelvisBoneFeatureBaryIndices, pelvisBoneFeatureBaryCoords);
                
                # Compute distances from predicted pelvis mesh to ground truth pelvis mesh
                # averageMeshDistance = sp.computeAveragePointsToPointsDistance(pdPelvisBoneMesh.vertices, gtBoneMesh.vertices);

                # Compute distances from predicted pelvis features to ground truth pelvis features
                # featureDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneFeatures, gtPelvisBoneFeatures);
                # averageFeatureDistance = np.mean(featureDistances);

                # Compute distances in muscle attachment points
                # nearestGtBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(pdPelvisBoneMesh.vertices, gtBoneMesh.vertices);
                # nearestGtBoneVertices = gtBoneMesh.vertices[nearestGtBoneVertexIndices];
                # correspondingDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMesh.vertices, nearestGtBoneVertices);
                # muscleAttachmentDistances = correspondingDistances[muscleAttachmentIndices];
                # averageMuscleAttachmentDistance = np.mean(muscleAttachmentDistances);

                # Compute vertex 2 vertex distances
                # nearestMeshVertices = sp.estimateNearestPointsFromPoints(pdPelvisBoneMesh.vertices, gtPersonalizedBoneVertices);
                # vertex2VertexDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMesh.vertices, nearestMeshVertices);

                # Compute vertex 2 vertex mesh distances
                # nearestMeshVertices = sp.estimateNearestPointsFromPoints(pdPelvisBoneMesh.vertices, gtBoneMesh.vertices);
                # vertex2VertexMeshDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMesh.vertices, nearestMeshVertices);

                # Compute vertex 2 vertex bone muscle distances
                nearestBoneMuscleVertices = sp.estimateNearestPointsFromPoints(pdPelvisBoneMuscleMesh.vertices, gtPersonalizedBoneMuscleMesh.vertices);
                vertex2VertexBoneMuscleDistances = sp.computeCorrespondingDistancesPoints2Points(pdPelvisBoneMuscleMesh.vertices, nearestBoneMuscleVertices);

                # Save the computed errors to buffers
                # meshTestingErrors[validIndex, testIndex] = averageMeshDistance;
                # featureTestingErrors[validIndex, testIndex] = averageFeatureDistance;
                # muscleAttachmentTestingErrors[validIndex, testIndex] = averageMuscleAttachmentDistance;
                # vertex2VertexDistanceBuffer.append(vertex2VertexDistances);
                # vertex2VertexMeshDistanceBuffer.append(vertex2VertexMeshDistances);
                vertex2VertexBoneMuscleDistanceBuffer.append(vertex2VertexBoneMuscleDistances);
        
        # Convert the buffer to numpy array
        print("Converting the buffer to numpy array ...");
        # vertex2VertexDistanceBuffer = np.array(vertex2VertexDistanceBuffer);
        # vertex2VertexMeshDistanceBuffer = np.array(vertex2VertexMeshDistanceBuffer);
        vertex2VertexBoneMuscleDistanceBuffer = np.array(vertex2VertexBoneMuscleDistanceBuffer);

        # Save the computed errors to files
        print("Saving the computed errors to files ...");
        # sp.saveMatrixToCSVFile(meshFeatureMuscleErrorFolder + f"/MeshTestingErrors_{featureSelIndex}.csv", meshTestingErrors);
        # sp.saveMatrixToCSVFile(meshFeatureMuscleErrorFolder + f"/FeatureTestingErrors_{featureSelIndex}.csv", featureTestingErrors);
        # sp.saveMatrixToCSVFile(meshFeatureMuscleErrorFolder + f"/MuscleAttachmentTestingErrors_{featureSelIndex}.csv", muscleAttachmentTestingErrors);
        # sp.saveNumPyArrayToNPY(optimalTrainValidTestFolder + f"/Vertex2VertexDistances_{featureSelIndex}.npy", vertex2VertexDistanceBuffer);
        # sp.saveNumPyArrayToNPY(meshFeatureMuscleErrorFolder + f"/Vertex2VertexMeshDistances_{featureSelIndex}.npy", vertex2VertexMeshDistanceBuffer);
        sp.saveNumPyArrayToNPY(meshFeatureMuscleErrorFolder + f"/Vertex2VertexBoneMuscleDistances_{featureSelIndex}.npy", vertex2VertexBoneMuscleDistanceBuffer);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_drawTestingErrorsInVariousMetrics():
    """
    Analyzes and visualizes testing errors from multi-output ridge regression pelvis reconstruction across various error metrics.
    
    This function evaluates the final testing performance of the optimal pelvis bone-muscle reconstruction model
    by loading testing error data across different metrics (mesh, feature, muscle attachment) and generating 
    comprehensive statistical visualizations to assess model accuracy and feature importance relationships.
    
    Key Steps:
    1. Initialize file paths and load feature selection strategy configurations
    2. Load testing error data for multiple error metrics across all feature strategies and validation folds
    3. Calculate the number of features for each feature selection strategy
    4. For each error metric type:
       - Aggregate testing errors and compute statistical measures (mean, standard deviation)
       - Generate bar chart visualization with error bars and trend analysis
       - Perform polynomial fitting to identify error trends with feature count
       - Calculate theoretical zero-error point for feature requirement estimation
       - Generate vertex-level color maps on 3D mesh models for spatial error visualization
    5. Save visualization results for publication and analysis
    
    Error Metrics Analyzed:
    - Mesh Testing Errors: Point-to-point distances between predicted and CT bone meshes
    - Feature Testing Errors: Accuracy of anatomical feature point reconstruction
    - Muscle Attachment Errors: Accuracy at critical muscle attachment locations
    - Vertex-to-Vertex Distances: Comprehensive vertex-level accuracy on personalized meshes
    - Vertex-to-Mesh Distances: Accuracy against CT-reconstructed bone surfaces
    - Vertex-to-BoneMuscle Distances: Overall bone-muscle structure accuracy
    
    Data Processing:
    - Reads testing errors from CSV and NPY files organized by feature strategy and validation fold
    - Processes errors across all available feature selection strategies (0 to max)
    - Computes mean and standard deviation across 10-fold cross-validation
    - Converts measurements from meters to millimeters for clinical presentation
    
    Statistical Analysis:
    - Mean testing error computation across cross-validation folds
    - Standard deviation calculation for error bar representation
    - Linear polynomial fitting to identify feature count vs error relationship
    - Zero-error point estimation to determine theoretical feature requirements
    
    Visualization Functions:
    - drawMeshTestingErrorsOnCTPelvisMeshes(): Bar chart analysis for CT mesh accuracy
    - drawFeatureTestingErrorsOnCTPelvisMeshes(): Bar chart analysis for feature point accuracy
    - drawMuscleAttachmentTestingErrorsOnCTPelvisMeshes(): Bar chart analysis for muscle attachment accuracy
    - drawVertex2VertexDistanceColorMapOnPersonalizedPelvisMeshes(): 3D color map on personalized meshes
    - drawVertex2VertexDistanceColorMapOnCTReconstructedPelvisMeshes(): 3D color map on CT meshes
    - drawVertex2VertexDistanceColorMapOnPelvisBoneMuscleMeshes(): 3D color map on bone-muscle meshes
    
    Visualization Outputs:
    - Bar charts showing mean testing errors with standard deviation error bars
    - Precise error value annotations on each bar for quantitative assessment
    - Polynomial trend lines with equations and zero-error point estimation
    - 3D color-coded mesh visualizations showing spatial error distribution
    - Color rulers and legends for error magnitude interpretation
    - Professional formatting with bold fonts and clear axis labeling
    
    Feature Analysis:
    - Maps feature selection strategy indices to actual feature counts
    - Analyzes relationship between number of anatomical features and reconstruction accuracy
    - Provides insights into optimal feature set size for clinical applications
    - Enables determination of minimum feature requirements for acceptable accuracy
    
    Output Files:
    - High-resolution PNG charts saved for publication use
    - 3D mesh visualizations with embedded error color maps
    - Statistical summary data for performance reporting
    - Trend analysis results for feature optimization guidance
    
    Note: This comprehensive function provides multiple perspectives on reconstruction accuracy,
    enabling detailed assessment of model performance across different anatomical regions and
    clinical applications. The combination of statistical charts and 3D visualizations offers
    both quantitative metrics and intuitive spatial understanding of prediction errors.
    """

    # Initializing
    print("Initializing ...");
    disk = "I:";
    mainFolder = disk + "/SpinalPelvisPred";
    pelvisReconFolder = mainFolder + "/Data/PelvisBoneRecon";
    crossValidationFolder = pelvisReconFolder + "/CrossValidation";
    featureSelectionProtocolFolder = mainFolder + "/Data/PelvisBoneRecon/CrossValidation/FeatureSelectionProtocol";
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    meshFeatureMuscleErrorFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/MeshFeatureMuscleErrors";
    optimalTrainValidTestFolder = crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest";
    numOfValids = 10; numOfTests = 30; numOfStrategies = len(featureSelectionStrategyDict);

    # Reading testing errors in various metrics
    print("Reading testing errors in various metrics ...");
    meshTestingErrors = np.zeros((numOfStrategies, numOfValids, numOfTests));
    featureTestingErrors = np.zeros((numOfStrategies, numOfValids, numOfTests));
    muscleAttachmentTestingErrors = np.zeros((numOfStrategies, numOfValids, numOfTests));
    for featureSelIndex in range(numOfStrategies):
        meshTestingErrors[featureSelIndex] = sp.readMatrixFromCSVFile(meshFeatureMuscleErrorFolder + f"/MeshTestingErrors_{featureSelIndex}.csv");
        featureTestingErrors[featureSelIndex] = sp.readMatrixFromCSVFile(meshFeatureMuscleErrorFolder + f"/FeatureTestingErrors_{featureSelIndex}.csv");
        muscleAttachmentTestingErrors[featureSelIndex] = sp.readMatrixFromCSVFile(meshFeatureMuscleErrorFolder + f"/MuscleAttachmentTestingErrors_{featureSelIndex}.csv");
    
    # Estimate the number of features in each feature selection strategy
    print("Estimating the number of features in each feature selection strategy ...");
    numOfStrategyFeatures = [];
    for featureSelIndex in range(numOfStrategies):
        featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
        numOfStrategyFeatures.append(len(featureSelectionIndices));
    numOfStrategyFeatures = np.array(numOfStrategyFeatures);
    
    # Draw the mesh testing errors.
    print("Draw the mesh testing errors ...");
    def drawMeshTestingErrorsOnCTPelvisMeshes():
        ## Forming drawing data
        meshMeans = np.mean(meshTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        meshStds = np.std(meshTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        ## Define label as the number of features in each feature selection strategy
        labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(numOfStrategies)];
        ## Set x locations and bar width
        xLocations = np.arange(numOfStrategies);
        barWidth = 0.75;
        ## Plotting the bar chart
        fig, ax = plt.subplots(figsize=(12, 6));
        meshBars = ax.bar(xLocations, meshMeans, width=barWidth, yerr=meshStds, capsize=5, label='Mesh Errors');
        ## Add value labels on top of each bar
        for bar, val in zip(meshBars, meshMeans):
            height = bar.get_height();
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom');
        ## Draw the poly fit line using linear regression line
        poly_fit = np.polyfit(xLocations, meshMeans, 1);
        poly_fit_line = np.polyval(poly_fit, xLocations);
        ## Draw the poly fit line
        line = ax.plot(xLocations, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
        ## Find the zero error point by fitting another line with real number of feature points
        realPolyFit = np.polyfit(numOfStrategyFeatures[xLocations], meshMeans, 1);
        ## Compute the zero error point when the mean error reach zero
        zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
        zero_error_y = np.polyval(realPolyFit, zero_error_x);
        ## Show the zero error point in the legend
        ax.legend([meshBars, line],
                [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
                loc='upper right', fontsize=12);
        ## Axis labels and ticks
        ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
        ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
        ax.set_xticks(xLocations);
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
        ax.set_yticks(ax.get_yticks());
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
        ## No grid
        ax.grid(False);
        ## Adding title
        ax.set_title('Testing Errors in CT-Reconstructed Pelvis Mesh', fontsize=16, fontweight='bold');
        ## Layout
        plt.tight_layout();
        plt.show();
        ## Save the figure
        fig.savefig(crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TestingErrorsInPelvisMesh.png", dpi=400, bbox_inches='tight');
    def drawFeatureTestingErrorsOnCTPelvisMeshes():
        ## Forming drawing data
        featureMeans = np.mean(featureTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        featureStds = np.std(featureTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        ## Define label as the number of features in each feature selection strategy
        labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(numOfStrategies)];
        ## Set x locations and bar width
        xLocations = np.arange(numOfStrategies);
        barWidth = 0.75;
        ## Plotting the bar chart
        fig, ax = plt.subplots(figsize=(12, 6));
        meshBars = ax.bar(xLocations, featureMeans, width=barWidth, yerr=featureStds, capsize=5, label='Mesh Errors');
        ## Add value labels on top of each bar
        for bar, val in zip(meshBars, featureMeans):
            height = bar.get_height();
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom');
        ## Draw the poly fit line using linear regression line
        poly_fit = np.polyfit(xLocations, featureMeans, 1);
        poly_fit_line = np.polyval(poly_fit, xLocations);
        ## Draw the poly fit line
        line = ax.plot(xLocations, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
        ## Find the zero error point by fitting another line with real number of feature points
        realPolyFit = np.polyfit(numOfStrategyFeatures[xLocations], featureMeans, 1);
        ## Compute the zero error point when the mean error reach zero
        zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
        zero_error_y = np.polyval(realPolyFit, zero_error_x);
        ## Show the zero error point in the legend
        ax.legend([meshBars, line],
                [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
                loc='upper right', fontsize=12);
        ## Axis labels and ticks
        ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
        ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
        ax.set_xticks(xLocations);
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
        ax.set_yticks(ax.get_yticks());
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
        ## No grid
        ax.grid(False);
        ## Adding title
        ax.set_title('Testing Errors in Pelvis Features', fontsize=16, fontweight='bold');
        ## Layout
        plt.tight_layout();
        plt.show();
        ## Save the figure
        fig.savefig(crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TestingErrorsInPelvisFeatures.png", dpi=400, bbox_inches='tight');
    def drawMuscleAttachmentTestingErrorsOnCTPelvisMeshes():
        ## Forming drawing data
        muscleMeans = np.mean(muscleAttachmentTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        muscleStds = np.std(muscleAttachmentTestingErrors, axis=(1, 2)) * 1000;  # Convert to mm
        ## Define label as the number of features in each feature selection strategy
        labels = [f"{len(list(featureSelectionStrategyDict.values())[i])}" for i in range(numOfStrategies)];
        ## Set x locations and bar width
        xLocations = np.arange(numOfStrategies);
        barWidth = 0.75;
        ## Plotting the bar chart
        fig, ax = plt.subplots(figsize=(12, 6));
        meshBars = ax.bar(xLocations, muscleMeans, width=barWidth, yerr=muscleStds, capsize=5, label='Mesh Errors');
        ## Add value labels on top of each bar
        for bar, val in zip(meshBars, muscleMeans):
            height = bar.get_height();
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom');
        ## Draw the poly fit line using linear regression line
        poly_fit = np.polyfit(xLocations, muscleMeans, 1);
        poly_fit_line = np.polyval(poly_fit, xLocations);
        ## Draw the poly fit line
        line = ax.plot(xLocations, poly_fit_line, color='red', linestyle='--', linewidth=2, label='Poly Fit Line')[0];
        ## Find the zero error point by fitting another line with real number of feature points
        realPolyFit = np.polyfit(numOfStrategyFeatures[xLocations], muscleMeans, 1);
        ## Compute the zero error point when the mean error reach zero
        zero_error_x = -realPolyFit[1] / realPolyFit[0];  # x when y = 0
        zero_error_y = np.polyval(realPolyFit, zero_error_x);
        ## Show the zero error point in the legend
        ax.legend([meshBars, line],
                [f'Mean Testing Errors (mm)', f'Poly Fit Line (y = {realPolyFit[0]:.3f}x + {realPolyFit[1]:.3f})\nZero Error Point: ({zero_error_x:.2f}, {zero_error_y:.2f})'],
                loc='upper right', fontsize=12);
        ## Axis labels and ticks
        ax.set_ylabel('Testing Errors (mm)', fontsize=14, fontweight='bold');
        ax.set_xlabel('Number of Features in Feature Selection Strategies', fontsize=14, fontweight='bold');
        ax.set_xticks(xLocations);
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold', ha='center');
        ax.set_yticks(ax.get_yticks());
        ax.set_yticklabels([f'{tick:.1f}' for tick in ax.get_yticks()], fontsize=13, fontweight='bold');
        ## No grid
        ax.grid(False);
        ## Adding title
        ax.set_title('Testing Errors in Muscle Attachment Points', fontsize=16, fontweight='bold');
        ## Layout
        plt.tight_layout();
        plt.show();
        ## Save the figure
        fig.savefig(crossValidationFolder + "/ShapeRelationStrategy/OptimalTrainValidTest/TestingErrorsInMuscleAttachmentPoints.png", dpi=400, bbox_inches='tight');
    def drawVertex2VertexDistanceColorMapOnPersonalizedPelvisMeshes():
        # Forming drawing data
        print("Reading testing errors for vertex to vertex distances ...");
        vertex2VertexDistances = sp.loadNumPYArrayFromNPY(optimalTrainValidTestFolder + f"/Vertex2VertexDistances_{20}.npy");

        # Reading template pelvis bone mesh for drawing color map
        print("Reading template pelvis bone mesh for drawing color map ...");
        templatePelvisBoneMesh = sp.readMesh(mainFolder + "/Data/Template/PelvisBonesMuscles/TempPelvisBoneMesh.ply");

        # Compute the average vertex to vertex distances
        print("Computing the average vertex to vertex distances ...");
        meanTestingError = np.mean(vertex2VertexDistances, axis=0);

        # Draw the color map of the testing errors as red (lowest) to violet (highest) on the vertex of the skull shape
        print("Drawing the color map of the testing errors ...");
        ## Normalize the mean testing error to [0, 1] range
        normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
        ## Create a custom colormap from red (low) to violet (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
        ## Map the normalized mean testing error to colors
        colors = colormap(normalizedMeanTestingError);
        ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
        templatePelvisBoneMesh.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
        trimeshViewer = trimesh.Scene(templatePelvisBoneMesh);
        trimeshViewer.show();

        # Generate and display color ruler
        print("Generating color ruler ...");
        ## Create a figure for the color ruler
        fig, ax = plt.subplots(figsize=(8, 2))    
        ## Create a gradient for the colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)    
        ## Display the gradient
        im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
        ## Calculate the actual error values for min, middle, and max
        min_error = np.min(meanTestingError) * 1000;  # Convert to mm
        max_error = np.max(meanTestingError) * 1000;  # Convert to mm
        mid_error = (min_error + max_error) / 2;
        ## Set up the colorbar ticks and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
        ax.set_yticks([])
        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Testing Vertex-to-Vertex Distances on Personalized Pelvis Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
        ## Add grid lines for better readability
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
        ## Adjust layout and display
        plt.tight_layout()
        ## Save the color ruler figure
        plt.savefig(optimalTrainValidTestFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
        ## Show the color ruler
        plt.show()

        # Finished processing
        print("Finished processing.");
    def drawVertex2VertexDistanceColorMapOnCTReconstructedPelvisMeshes():
        # Forming drawing data
        print("Reading testing errors for vertex to vertex distances ...");
        vertex2VertexDistances = sp.loadNumPYArrayFromNPY(optimalTrainValidTestFolder + f"/Vertex2VertexMeshDistances_{20}.npy");

        # Reading template pelvis bone mesh for drawing color map
        print("Reading template pelvis bone mesh for drawing color map ...");
        templatePelvisBoneMesh = sp.readMesh(mainFolder + "/Data/Template/PelvisBonesMuscles/TempPelvisBoneMesh.ply");

        # Compute the average vertex to vertex distances
        print("Computing the average vertex to vertex distances ...");
        meanTestingError = np.mean(vertex2VertexDistances, axis=0);

        # Draw the color map of the testing errors as red (lowest) to violet (highest) on the vertex of the skull shape
        print("Drawing the color map of the testing errors ...");
        ## Normalize the mean testing error to [0, 1] range
        normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
        ## Create a custom colormap from red (low) to violet (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
        ## Map the normalized mean testing error to colors
        colors = colormap(normalizedMeanTestingError);
        ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
        templatePelvisBoneMesh.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
        trimeshViewer = trimesh.Scene(templatePelvisBoneMesh);
        trimeshViewer.show();

        # Generate and display color ruler
        print("Generating color ruler ...");
        ## Create a figure for the color ruler
        fig, ax = plt.subplots(figsize=(8, 2))    
        ## Create a gradient for the colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)    
        ## Display the gradient
        im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
        ## Calculate the actual error values for min, middle, and max
        min_error = np.min(meanTestingError) * 1000;  # Convert to mm
        max_error = np.max(meanTestingError) * 1000;  # Convert to mm
        mid_error = (min_error + max_error) / 2;
        ## Set up the colorbar ticks and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
        ax.set_yticks([])
        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Testing Vertex-to-Vertex Distances on CT-reconstructed Pelvis Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
        ## Add grid lines for better readability
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
        ## Adjust layout and display
        plt.tight_layout()
        ## Save the color ruler figure
        plt.savefig(optimalTrainValidTestFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
        ## Show the color ruler
        plt.show()

        # Finished processing
        print("Finished processing.");
    def drawVertex2VertexDistanceColorMapOnPelvisBoneMuscleMeshes():
        # Forming drawing data
        print("Reading testing errors for vertex to vertex distances ...");
        vertex2VertexDistances = sp.loadNumPYArrayFromNPY(optimalTrainValidTestFolder + f"/Vertex2VertexBoneMuscleDistances_{20}.npy");

        # Reading template pelvis bone mesh for drawing color map
        print("Reading template pelvis bone mesh for drawing color map ...");
        templatePelvisBoneMuscleMesh = sp.readMesh(mainFolder + "/Data/Template/PelvisBonesMuscles/TempPelvisBoneMuscles.ply");

        # Compute the average vertex to vertex distances
        print("Computing the average vertex to vertex distances ...");
        meanTestingError = np.mean(vertex2VertexDistances, axis=0);

        # Draw the color map of the testing errors as red (lowest) to violet (highest) on the vertex of the skull shape
        print("Drawing the color map of the testing errors ...");
        ## Normalize the mean testing error to [0, 1] range
        normalizedMeanTestingError = (meanTestingError - np.min(meanTestingError)) / (np.max(meanTestingError) - np.min(meanTestingError));
        ## Create a custom colormap from red (low) to violet (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        colormap = LinearSegmentedColormap.from_list('red_to_violet', colors_list)
        ## Map the normalized mean testing error to colors
        colors = colormap(normalizedMeanTestingError);
        ## Set the colors to the vertices of the template pelvis bone mesh using trimesh and visualize it
        templatePelvisBoneMuscleMesh.visual.vertex_colors = colors[:, :3] * 255;  # Convert to 0-255 range
        trimeshViewer = trimesh.Scene(templatePelvisBoneMuscleMesh);
        trimeshViewer.show();

        # Generate and display color ruler
        print("Generating color ruler ...");
        ## Create a figure for the color ruler
        fig, ax = plt.subplots(figsize=(8, 2))    
        ## Create a gradient for the colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)    
        ## Display the gradient
        im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, 0, 1])    
        ## Calculate the actual error values for min, middle, and max
        min_error = np.min(meanTestingError) * 1000;  # Convert to mm
        max_error = np.max(meanTestingError) * 1000;  # Convert to mm
        mid_error = (min_error + max_error) / 2;
        ## Set up the colorbar ticks and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([f'{min_error:.2f} mm\n(Min)', f'{mid_error:.2f} mm\n(Middle)', f'{max_error:.2f} mm\n(Max)'])
        ax.set_yticks([])
        ax.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
        ax.set_title('Testing Vertex-to-Vertex Distances on Personalized Pelvis Bone Muscle Meshes\n(Mean ± Std = {:.2f} ± {:.2f} mm)'.format(np.mean(meanTestingError) * 1000, np.std(meanTestingError) * 1000), fontsize=14, fontweight='bold')    
        ## Add grid lines for better readability
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)    
        ## Adjust layout and display
        plt.tight_layout()
        ## Save the color ruler figure
        plt.savefig(optimalTrainValidTestFolder + "/ColorMapLegend_PelvisPredictionErrors.png", bbox_inches='tight', dpi=300)
        ## Show the color ruler
        plt.show()

        # Finished processing
        print("Finished processing.");
    
    # Running the functions
    print("Running the functions ...");
    drawMuscleAttachmentTestingErrorsOnCTPelvisMeshes();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_estimateBestAndWorstPredictedCases():
    """
    Identifies the best and worst performing test cases from multi-output ridge regression pelvis reconstruction results.
    
    This function analyzes testing error data across all validation folds to identify the specific test cases
    that achieved the lowest (best) and highest (worst) reconstruction errors. This analysis helps in understanding
    model performance extremes and enables targeted case study analysis for method improvement.
    
    Key Steps:
    1. Load testing error data from all validation folds using feature selection strategy 20 (38 features)
    2. Aggregate testing errors across multiple cross-validation folds into a single matrix
    3. Identify the global minimum and maximum error values across all test cases
    4. Determine the validation fold and subject indices corresponding to best/worst performance
    5. Map these indices to actual subject IDs using cross-validation split files
    6. Report the specific cases for detailed analysis and visualization
    
    Data Processing:
    - Reads testing errors from CSV files organized by validation fold
    - Processes 10 validation folds with 30 test subjects each (300 total test cases)
    - Uses feature selection strategy 20 which contains 38 anatomical feature points
    - Reshapes error matrix for efficient global minimum/maximum identification
    
    Analysis Strategy:
    - Global optimization across all test cases to find performance extremes
    - Cross-validation fold mapping to maintain data traceability
    - Subject ID resolution for case-specific analysis and visualization
    
    Output Information:
    - Best case: Validation fold index, subject index, error value, and subject ID
    - Worst case: Validation fold index, subject index, error value, and subject ID
    - Enables targeted analysis of model performance characteristics
    - Provides specific cases for detailed visualization and error analysis
    
    Statistical Summary:
    - Reports minimum and maximum reconstruction errors across all test cases
    - Identifies specific validation configurations that produced extreme results
    - Enables understanding of model performance distribution and outliers
    
    Use Cases:
    - Performance analysis: Understanding model capabilities and limitations
    - Case study selection: Identifying representative examples for detailed analysis
    - Method improvement: Analyzing failure modes and success patterns
    - Visualization preparation: Selecting cases for 3D mesh comparison and error mapping
    
    Note: This function focuses on feature selection strategy 20 (38 features) as it represents
    an optimal balance between feature richness and reconstruction accuracy based on validation results.
    The identified cases can be used for detailed visualization, error analysis, and method comparison.
    """

    # Initializing
    print("Initializing ...");
    disk = "I:";
    testingErrorFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\TestingErrors";
    trainTestSplitFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\TrainTestSplits";
    numOfValids = 10;

    # Reading testing errors in various validation with the feature of 38
    print("Reading testing errors in various validation with the feature of 38 ...");
    testingErrors = [];
    for v in range(numOfValids):
        testingErrorFilePath = testingErrorFolder + f"/AveragePoint2PointDistances_20_{v}.csv";
        testingError = sp.readMatrixFromCSVFile(testingErrorFilePath);
        testingErrors.append(testingError);
    testingErrors = np.array(testingErrors);
    testingErrors = testingErrors.reshape((numOfValids, -1));
    print("\t The shape of testingErrors: ", testingErrors.shape);

    # Estimate the best and worst predicted cases
    print("Estimating the best and worst predicted cases ...");
    min_flat_index = np.argmin(testingErrors);
    max_flat_index = np.argmax(testingErrors);
    min_i, min_j = np.unravel_index(min_flat_index, testingErrors.shape)
    max_i, max_j = np.unravel_index(max_flat_index, testingErrors.shape)

    # Report the values for finding the indices
    print("Report the values ...");
    print(f"Minimum testing error: {testingErrors[min_i, min_j]} (Validation {min_i}, Subject Index {min_j})");
    print(f"Maximum testing error: {testingErrors[max_i, max_j]} (Validation {max_i}, Subject Index {max_j})");

    # Determing the best and worst ID
    print("Determinging the best and worst ID ...");
    minTestIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TestingIDs_{min_i}.txt");
    maxTestIDs = sp.readListOfStrings(trainTestSplitFolder + f"/TestingIDs_{max_i}.txt");
    minTestID = minTestIDs[min_j];
    maxTestID = maxTestIDs[max_j];
    print("\t The best predicted ID: ", minTestID);
    print("\t The worst predicted ID: ", maxTestID);

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCases():
    """
    Visualizes and compares the best and worst performing pelvis reconstruction cases from multi-output ridge regression results.
    
    This function provides detailed visualization and error analysis of the extreme performance cases identified
    from pelvis bone-muscle reconstruction experiments. It loads predicted meshes, ground truth data, and performs
    comprehensive error calculations with 3D visualization to understand model performance characteristics.
    
    Key Steps:
    1. Initialize file paths for best and worst predicted case data storage
    2. Load template pelvis bone-muscle mesh information and compute vertex mappings
    3. Load predicted and ground truth meshes for both best and worst performing cases:
       - Best case: Validation fold 0, subject index 17 (Subject ID: 257)
       - Worst case: Validation fold 5, subject index 19 (Subject ID: 1091)
    4. Perform rigid registration between predicted and personalized ground truth meshes
    5. Extract bone-only meshes from bone-muscle structures for CT comparison
    6. Compute comprehensive error metrics for both cases:
       - Predicted vs personalized bone-muscle mesh distances
       - Predicted vs CT bone mesh distances
       - Statistical measures (mean, standard deviation) in millimeters
    7. Generate 3D visualizations with color-coded meshes for comparative analysis
    
    Data Processing:
    - Uses pre-identified best/worst cases from feature selection strategy 20 (38 features)
    - Applies rigid SVD transformation for fair coordinate system alignment
    - Extracts bone vertices from bone-muscle structures for targeted comparison
    - Computes nearest point correspondences for accurate distance measurements
    
    Error Analysis:
    - Point-to-point distance calculations between predicted and ground truth meshes
    - Separate analysis for personalized (bone-muscle) and CT (bone-only) comparisons
    - Statistical reporting with mean and standard deviation in clinically relevant units
    - Quantitative assessment of reconstruction quality extremes
    
    Visualization Features:
    - Color-coded mesh rendering with distinct colors for predicted vs ground truth
    - Red coloring for predicted meshes to highlight reconstruction results
    - Bone-colored rendering for ground truth meshes for anatomical realism
    - Interactive 3D scene viewer for detailed inspection and comparison
    - Side-by-side visualization capability for direct comparison
    
    Output Metrics (Example Results):
    - Best case performance: Lower reconstruction errors demonstrating model capability
    - Worst case performance: Higher reconstruction errors indicating model limitations
    - Statistical summaries enable quantitative performance range assessment
    - Visual inspection reveals spatial error patterns and anatomical accuracy
    
    Use Cases:
    - Performance validation: Understanding model capability range and limitations
    - Case study analysis: Detailed examination of success and failure patterns
    - Method improvement: Identifying areas for algorithmic enhancement
    - Clinical assessment: Evaluating reconstruction quality for medical applications
    - Publication visualization: Generating figures for research communication
    
    Note: This function focuses on the extreme cases from a comprehensive evaluation
    using 38 anatomical features, providing insights into the best achievable accuracy
    and potential failure modes of the multi-output ridge regression approach for
    pelvis reconstruction from anatomical landmarks.
    """

    # Initialize
    print("Initializing ...");
    bestWorstPredictedFolder = r"I:\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\BestWorstPredictedCases";
    bestPredictedFolder = bestWorstPredictedFolder + "/BestPredictedCase";
    worstPredictedFolder = bestWorstPredictedFolder + "/WorstPredictedCase";
    templateFolder = r"I:\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";

    # Reading the template information
    print("Reading the template information ...");
    templatePelvisBoneMuscleMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMuscles.ply");
    templatePelvisBoneMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMesh.ply");
    pelvisBoneVertexIndices = sp.estimateNearestIndicesKDTreeBased(templatePelvisBoneMesh.vertices, templatePelvisBoneMuscleMesh.vertices);

    # Reading information
    print("Reading information ...");
    bestPredictedBoneMuscleMesh = sp.readMesh(bestPredictedFolder + "/PredictedPelvicMuscle_20_0_17.ply");
    bestPersonalizedBoneMuscleMesh = sp.readMesh(bestPredictedFolder + "/257-PersonalizedPelvisBoneMuscleMesh.ply");
    bestGroundTruthBoneMesh = sp.readMesh(bestPredictedFolder + "/257-PelvisBoneMesh.ply");
    worstPredictedBoneMuscleMesh = sp.readMesh(worstPredictedFolder + "/PredictedPelvicMuscle_20_5_19.ply");
    worstPersonalizedBoneMuscleMesh = sp.readMesh(worstPredictedFolder + "/1091-PersonalizedPelvisBoneMuscleMesh.ply");
    worstGroundTruthBoneMesh = sp.readMesh(worstPredictedFolder + "/1091-PelvisBoneMesh.ply");

    # Registration the predicted mesh to the personalized mesh
    print("Registration the predicted mesh to the personalized mesh ...");
    bestSVDTransform = sp.estimateRigidSVDTransform(bestPredictedBoneMuscleMesh.vertices, bestPersonalizedBoneMuscleMesh.vertices);
    worstSVDTransform = sp.estimateRigidSVDTransform(worstPredictedBoneMuscleMesh.vertices, worstPersonalizedBoneMuscleMesh.vertices);
    bestPredictedBoneMuscleMesh = sp.transformMesh(bestPredictedBoneMuscleMesh, bestSVDTransform);
    worstPredictedBoneMuscleMesh = sp.transformMesh(worstPredictedBoneMuscleMesh, worstSVDTransform);
    bestPredictedPelvisBoneMesh = sp.cloneMesh(templatePelvisBoneMesh);
    worstPPredictedPelvisBoneMesh = sp.cloneMesh(templatePelvisBoneMesh);
    bestPredictedPelvisBoneMesh.vertices = bestPredictedBoneMuscleMesh.vertices[pelvisBoneVertexIndices];
    worstPPredictedPelvisBoneMesh.vertices = worstPredictedBoneMuscleMesh.vertices[pelvisBoneVertexIndices];

    # Computing some errors for illustration
    print("Computing some errors for illustration ...");
    ## Compute best bone muscle errors
    bestPersonalizedNearestVertices = sp.estimateNearestPointsFromPoints(bestPredictedBoneMuscleMesh.vertices, bestPersonalizedBoneMuscleMesh.vertices);
    bestPersonalizedDistances = sp.computeCorrespondingDistancesPoints2Points(bestPersonalizedBoneMuscleMesh.vertices, bestPersonalizedNearestVertices);
    bestPersonalizedMeanDistances = np.mean(bestPersonalizedDistances)*1000;
    bestPersonalizedStdDistances = np.std(bestPersonalizedDistances)*1000;
    print("\t Best personalized Mean +- SD: ", f"{bestPersonalizedMeanDistances} +- {bestPersonalizedStdDistances}");
    ## Compute best bone errors
    bestGroundTruthNearestVertices = sp.estimateNearestPointsFromPoints(bestPredictedPelvisBoneMesh.vertices, bestGroundTruthBoneMesh.vertices);
    bestGroundTruthDistances = sp.computeCorrespondingDistancesPoints2Points(bestPredictedPelvisBoneMesh.vertices, bestGroundTruthNearestVertices);
    bestGroundTruthMeanDistances = np.mean(bestGroundTruthDistances)*1000;
    bestGroundTruthStdDistances = np.std(bestGroundTruthDistances)*1000;
    print("\t Best ground truth Mean +- SD: ", f"{bestGroundTruthMeanDistances} +- {bestGroundTruthStdDistances}");
    ## Compute worst bone muscle errors
    worstPersonalizedNearestVertices = sp.estimateNearestPointsFromPoints(worstPredictedBoneMuscleMesh.vertices, worstPersonalizedBoneMuscleMesh.vertices);
    worstPersonalizedDistances = sp.computeCorrespondingDistancesPoints2Points(worstPredictedBoneMuscleMesh.vertices, worstPersonalizedNearestVertices);
    worstPersonalizedMeanDistances = np.mean(worstPersonalizedDistances)*1000;
    worstPersonalizedStdDistances = np.std(worstPersonalizedDistances)*1000;
    print("\t Worst personalized Mean +- SD: ", f"{worstPersonalizedMeanDistances} +- {worstPersonalizedStdDistances}");
    ## Compute worst bone errors
    worstGroundTruthNearestVertices = sp.estimateNearestPointsFromPoints(worstPPredictedPelvisBoneMesh.vertices, worstGroundTruthBoneMesh.vertices);
    worstGroundTruthDistances = sp.computeCorrespondingDistancesPoints2Points(worstPPredictedPelvisBoneMesh.vertices, worstGroundTruthNearestVertices);
    worstGroundTruthMeanDistances = np.mean(worstGroundTruthDistances)*1000;
    worstGroundTruthStdDistances = np.std(worstGroundTruthDistances)*1000;
    print("\t Worst ground truth Mean +- SD: ", f"{worstGroundTruthMeanDistances} +- {worstGroundTruthStdDistances}");

    # Visualize the mesh using the trimesh libraries
    print("Visualize the mesh using the trimesh library ...");
    ## Fill color for the predicted meshes as red color
    bestPredictedBoneMuscleMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    worstPredictedBoneMuscleMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    ## Fill the color of the personalized pelvis bone mesh as the bone color
    bestPersonalizedBoneMuscleMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    worstPersonalizedBoneMuscleMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    ## Fill the color of predicted pelvis bone mesh as red color
    bestPredictedPelvisBoneMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    worstPPredictedPelvisBoneMesh.visual.vertex_colors = [255, 0, 0, 255];  # Red color
    ## Fill the ground truth pelvis bone mesh as the bone color
    bestGroundTruthBoneMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    worstGroundTruthBoneMesh.visual.vertex_colors = [227, 218, 201, 200];  # White color
    ## Visualize the predicted bone muscle with the personalized bone muscles using the trimesh 
    scene = trimesh.Scene([worstPPredictedPelvisBoneMesh, worstGroundTruthBoneMesh]);
    scene.show();

    # Finished processing
    print("Finished processing.");
def featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCasesWithCTImageScans():
    """
    Visualizes and compares pelvis reconstruction results with CT image scans for comprehensive clinical assessment.
    
    This function provides advanced visualization capabilities by overlaying predicted pelvis bone-muscle structures
    onto actual CT image scans, enabling direct comparison between computational predictions and medical imaging data.
    The visualization combines 3D mesh reconstruction results with volumetric CT data for clinical validation.
    
    Key Steps:
    1. Parse command line argument for feature selection strategy index
    2. Initialize file paths for DICOM illustration data and template information
    3. Load template pelvis bone-muscle mesh and feature selection strategy configurations
    4. Load predicted pelvis bone-muscle mesh and corresponding anatomical features
    5. Load CT-reconstructed spino-pelvic bone mesh and torso structures
    6. Perform rigid registration between predicted and CT coordinate systems
    7. Scale meshes to match DICOM image coordinate system (millimeter units)
    8. Extract structural feature points from predicted mesh using barycentric coordinates
    9. Generate 3D sphere markers for anatomical feature visualization
    10. Load and process DICOM image series for volumetric rendering
    11. Create interactive 3D visualization combining:
        - Volumetric CT image rendering with tissue-specific opacity settings
        - Predicted pelvis bone-muscle mesh overlays
        - Anatomical feature point markers
        - Ground truth CT bone structures for comparison
    
    Command Line Arguments:
    - FeatureSelIndex: Index specifying which feature selection strategy to visualize
    
    Data Processing:
    - Uses ITK for DICOM image series reading and processing
    - Applies rigid SVD transformation for coordinate system alignment
    - Scales all 3D structures from meters to millimeters for DICOM compatibility
    - Reconstructs anatomical features using barycentric landmark interpolation
    
    Registration and Alignment:
    - Registers predicted pelvis to CT-reconstructed reference using feature correspondences
    - Ensures spatial alignment between computational predictions and medical imaging
    - Maintains anatomical accuracy through rigid transformation constraints
    
    Visualization Features:
    - Volumetric CT rendering with tissue-differentiated opacity:
      * Background: Completely transparent (opacity = 0.0)
      * Soft tissue: Very low opacity (opacity = 0.05) for context
      * Bone tissue: Moderate opacity (opacity = 0.3) for structural visibility
    - Color-coded anatomical structures:
      * CT bone structures: Natural bone color (RGB: 227, 218, 201)
      * Predicted structures: Overlaid for direct comparison
      * Feature points: Red spherical markers for landmark identification
    - Interactive 3D navigation with trackball camera controls
    - Professional medical imaging color schemes and opacity settings
    
    Technical Implementation:
    - ITK integration for medical image processing and DICOM handling
    - VTK pipeline for advanced 3D visualization and volume rendering
    - Trimesh to VTK conversion for mesh structure compatibility
    - Smart volume mapping with tissue-specific transfer functions
    
    Clinical Applications:
    - Validation of computational pelvis reconstruction against medical imaging
    - Assessment of prediction accuracy in clinical coordinate systems
    - Visual verification of anatomical feature correspondence
    - Quality control for reconstruction algorithms in medical applications
    - Pre-surgical planning and biomechanical analysis support
    
    Output Capabilities:
    - Interactive 3D scene with real-time manipulation
    - Simultaneous visualization of predicted and actual anatomical structures
    - Anatomical landmark verification through feature point overlays
    - Clinical-grade visualization suitable for medical assessment
    
    Use Cases:
    - Clinical validation of reconstruction algorithms
    - Medical imaging research and algorithm development
    - Biomechanical modeling verification
    - Surgical planning assistance
    - Educational demonstration of computational anatomy methods
    
    Note: This function represents the culmination of the reconstruction pipeline,
    providing clinical-grade visualization that enables direct assessment of
    computational predictions against medical imaging gold standards. The integration
    of CT imaging with mesh reconstruction offers comprehensive validation capabilities
    essential for medical applications.
    """
    
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("Please input the command as: [ProgramName] [FeatureSelIndex]");
        return
    featureSelIndex = int(sys.argv[1]);
    disk = "I:";
    mainFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon";
    dicomIllustrationFolder = mainFolder + r"\CrossValidation\ShapeRelationStrategy\OptimalTrainValidTest\DICOMIllustration";
    templateFolder = disk + r"\SpinalPelvisPred\Data\Template\PelvisBonesMuscles";
    featureSelectionProtocolFolder = disk + r"\SpinalPelvisPred\Data\PelvisBoneRecon\CrossValidation\FeatureSelectionProtocol";

    # Reading template data
    print("Reading template data ...");
    templateBoneMuscleMesh = sp.readMesh(templateFolder + "/TempPelvisBoneMuscles.ply");
    allPelvicFeatures = sp.read3DPointsFromPPFile(featureSelectionProtocolFolder + "/AllFeaturePoints.pp");
    featureSelectionStrategyDict = sp.readVectorsFromFile(featureSelectionProtocolFolder + "/FeatureSelectionIndexStrategies.txt");
    featureSelectionIndices = list(featureSelectionStrategyDict.values())[featureSelIndex];
    tempPelvicFeatures = allPelvicFeatures[featureSelectionIndices];
    pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = sp.computeBarycentricLandmarks(templateBoneMuscleMesh, tempPelvicFeatures);

    # Reading source and target meshes
    print("Reading predicted case ...");
    predictedPelvicBoneMuscleMesh = sp.readMesh(dicomIllustrationFolder + "/PredictedPelvicMuscle_20_0_0.ply");
    predictedPelvicBoneMuscleFeatures = sp.read3DPointsFromPPFile(dicomIllustrationFolder + "/PredictedPelvicMuscle_20_0_0_picked_points.pp");
    ctSpinoPelvicBoneMesh = sp.readMesh(dicomIllustrationFolder + "/138645-SpinopelvicBoneMesh.ply");
    ctPelvicBoneFeatures = sp.read3DPointsFromPPFile(dicomIllustrationFolder + "/138645-SpinopelvicBoneMesh_picked_points.pp");
    ctTorsoMesh = sp.readMesh(dicomIllustrationFolder + "/138645-TorsoMesh_Simplified.ply");
    
    # Register the predicted pelvic bone muscle mesh to the target pelvic mesh
    print("Register the predicted to the target ...");
    ## Estimate the svd transform
    svdTransform = sp.estimateRigidSVDTransform(predictedPelvicBoneMuscleFeatures, ctPelvicBoneFeatures);
    ## Transform the predicted pelvis bone mesh to the 
    predictedPelvicBoneMuscleMesh = sp.transformMesh(predictedPelvicBoneMuscleMesh, svdTransform);
    ## Scale the meshes to the scale of the DICOM images
    ctSpinoPelvicBoneMesh.vertices *= 1000;
    predictedPelvicBoneMuscleMesh.vertices *= 1000;
    ctTorsoMesh.vertices *= 1000;
    ## Colorize the vertice of the ctSpinoPelvicBoneMesh as bone color rgb
    boneColorRGBA = np.array([227, 218, 201, 200], dtype=np.uint8);
    ctSpinoPelvicBoneMesh.visual.vertex_colors = np.tile(boneColorRGBA, (len(ctSpinoPelvicBoneMesh.vertices), 1));
    ## Compute the structural features from the predicted pelvic bone muscle mesh
    structuralFeatures = sp.reconstructPointsFromBaryCentric(predictedPelvicBoneMuscleMesh, pelvicFeatureBaryCoords, pelvicFeatureBaryIndices);
    ## Generate the structural feature points
    structuralFeatureSpheres = [];
    for i in range(len(structuralFeatures)):
        coord = structuralFeatures[i];
        sphere = sp.generateSphereMesh(inCenter=coord, inRadius=10, inResolution=2);
        structuralFeatureSpheres.append(sphere);

    # Visualize the dicom images of the CT
    print("Visualize the dicom image of the CT ...");
    dicomFolder = dicomIllustrationFolder + "/DICOM";
    ## Reading the dicom series
    print("\t\t Reading the dicom series ...");
    import itk;
    ImageType = itk.Image[itk.SS, 3]  # Signed short, 3D
    reader = itk.ImageSeriesReader[ImageType].New()
    names_generator = itk.GDCMSeriesFileNames.New()
    names_generator.SetDirectory(dicomFolder)
    dicom_names = names_generator.GetInputFileNames()
    reader.SetFileNames(dicom_names)
    reader.Update()
    image = reader.GetOutput()
    ## Convert itk image to vtk image
    print("\t\t Convert itk image to vtk image ...");
    vtk_image = itk.vtk_image_from_image(image)
    ## Visualize using vtk
    print("\t\t Visualize using vtk ...");
    import vtk;
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.0)      # Background - completely transparent
    opacity.AddPoint(500, 0.05)   # Soft tissue - very low opacity (reduced from 0.15)
    opacity.AddPoint(1000, 0.3)   # Bone - reduced opacity (reduced from 0.85)
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color.AddRGBPoint(500, 1.0, 1.0, 0.9)
    color.AddRGBPoint(1000, 1.0, 1.0, 1.0)
    volume_property.SetScalarOpacity(opacity)
    volume_property.SetColor(color)
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(1.0, 1.0, 1.0)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interactor_style)
    ## Add more the structural feature points to the renderer
    for i, sphere_mesh in enumerate(structuralFeatureSpheres):
        # Convert trimesh to VTK polydata
        vtk_points = vtk.vtkPoints()
        for vertex in sphere_mesh.vertices:
            vtk_points.InsertNextPoint(vertex[0], vertex[1], vertex[2])        
        vtk_cells = vtk.vtkCellArray()
        for face in sphere_mesh.faces:
            vtk_cells.InsertNextCell(3)
            for vertex_id in face:
                vtk_cells.InsertCellPoint(vertex_id)        
        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)        
        # Set sphere color (e.g., red for visibility)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color
        actor.GetProperty().SetOpacity(1.0)  # Fully opaque
        # Add actor to renderer
        renderer.AddActor(actor)
    ## Render the spheres
    interactor.Initialize()
    interactor.Start()

    # Finished processing
    print("Finished processing.");

#**************************************** MAIN FUNCTION
def main():
    os.system("cls");
    featureToPelvisStructureRecon_shapeRelationStrategy_BoneAndMuscleStructures_MultiOutputRegressor_visualizeBestWorstPredictedCasesWithCTImageScans();
if __name__ == "__main__":
    main()
