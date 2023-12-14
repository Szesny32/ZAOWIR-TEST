import LAB1
import Stereo
import Calibration
import cv2 as cv
import Tools
import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import copy
import msvcrt 

CHECKBOARD_PATTERN = (8, 6)
WINDOW_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
PATH = "datasets/Lab1/s1/"
extension =".png"
prefix_L, prefix_R = ("left_", "right_")
WORKSPACE = {
    'corners_L':            'resources/corners_L.json',
    'subcorners_L' :        'resources/subcorners_L.json',
    'corners_R':            'resources/corners_R.json',
    'subcorners_R' :        'resources/subcorners_R.json',

    'calibration_L':        'resources/calibration_corners_L.json',
    'sub_calibration_L':    'resources/calibration_sub_corners_L.json',
    'calibration_R':        'resources/calibration_corners_R.json',
    'sub_calibration_R':    'resources/calibration_sub_corners_R.json',

    'disortImage' :         'datasets/Lab1/s1/left_4.png',
    'undisortImage':        'resources/undisort.png',
    'stereoConfigFile':     'resources/stereo_configuration.json'
}

CALIBRATION_SAMPLES = 50
CHECKBOARD_SIZE = 2.867
example = "datasets/Lab1/s1/left_0.png"
( _height, _width) = cv.imread("datasets/Lab1/s1/left_0.png").shape[:2]

def task1():
    (_, _, subcorners_dataset_L, subcorners_dataset_R) = LAB1.task1_result()
    dataset_metadata = ( "datasets/Lab1/s1/",  "left_",   "right_",  ".png") 
   
    (associated_L, associated_R) = Stereo.AssociateFrames(subcorners_dataset_L, subcorners_dataset_R, dataset_metadata, CALIBRATION_SAMPLES)
    corners_L = [corners for _, corners in associated_L]
    corners_R = [corners for _, corners in associated_R]

    calibration_L = Calibration.Calibrate(imgPoints = corners_L, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , imageSize= ( _height, _width))
    calibration_R = Calibration.Calibrate(imgPoints = corners_R, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , imageSize= ( _height, _width))
    stereoData = Stereo.Calibrate(corners_L, corners_R, calibration_L, calibration_R, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , imageSize= ( _height, _width))
    Tools.ExportStereo(WORKSPACE['stereoConfigFile'], stereoData)


def task2():
    stereoData = Tools.ImportStereo(WORKSPACE['stereoConfigFile'])
    (retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) =  stereoData
    baseline = np.linalg.norm(translationVec) 
    print(f"Odległość bazowa (baseline): {baseline:.2f} cm")

def task3_4():
    stereoData = Tools.ImportStereo(WORKSPACE['stereoConfigFile'])
    (retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) =  stereoData
    image_L = cv.imread(PATH + "left_3.png")
    image_R = cv.imread(PATH + "right_3.png")
    

    gray = cv.cvtColor(image_L, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify( cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, rotationMat, translationVec, flags=cv.CALIB_ZERO_DISPARITY, alpha=-1)

    map1_L, map2_L = cv.initUndistortRectifyMap( cameraMatrix1, distCoeffs1,R1, P1, image_size, cv.CV_32FC1)
    map1_R, map2_R = cv.initUndistortRectifyMap( cameraMatrix2, distCoeffs2, R2, P2, image_size, cv.CV_32FC1)

    #Zadanie 4 - Porównaj czas obliczeń oraz oceń subiektywnie wyniki dla różnych metod interpolacji
    interpolations = [cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA, cv.INTER_LANCZOS4]
    translation = ['INTER_NEAREST', 'INTER_LINEAR','INTER_CUBIC', 'INTER_AREA', 'INTER_LANCZOS4']
    for interpolation in interpolations:
        timestamp = time()
        undistorted_L = cv.remap(copy(image_L), copy(map1_L), copy(map2_L), interpolation)
        undistorted_R = cv.remap(copy(image_R), copy(map1_R), copy(map2_R), interpolation)
        print(f"Zakończono remap() dla interpolacji {translation[interpolation]} w czasie {(time() - timestamp):.4f}s\n")
        plt.subplot(2, 2, 1), plt.imshow(Tools.ScaleImage(image_L))
        plt.subplot(2, 2, 3), plt.imshow(Tools.ScaleImage(undistorted_L))
        plt.subplot(2, 2, 2), plt.imshow(Tools.ScaleImage(image_R))
        plt.subplot(2, 2, 4), plt.imshow(Tools.ScaleImage(undistorted_R))
        plt.show()



def task5_6():
    stereoData = Tools.ImportStereo(WORKSPACE['stereoConfigFile'])
    (retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) =  stereoData


    image_L = cv.imread(PATH + "left_3.png")
    image_R = cv.imread(PATH + "right_3.png")
    
    gray = cv.cvtColor(image_L, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify( cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, rotationMat, translationVec, flags=cv.CALIB_ZERO_DISPARITY, alpha=-1)

    map1_L, map2_L = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv.CV_32FC1)
    map1_R, map2_R = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv.CV_32FC1)

    # Zastosuj mapy rektyfikacji do obu obrazów
    rectified_img_L = cv.remap(image_L, map1_L, map2_L, cv.INTER_LINEAR)
    rectified_img_R = cv.remap(image_R, map1_R, map2_R, cv.INTER_LINEAR)
    image_L = Tools.ScaleImage(image_L)
    image_R = Tools.ScaleImage(image_R)
    rectified_img_L = Tools.ScaleImage(rectified_img_L)
    rectified_img_R =Tools.ScaleImage(rectified_img_R)
    # Wyświetl zrektyfikowane obrazy obok siebie
    cv.imshow('Zrektyfikowany obraz lewy', rectified_img_L)
    cv.imshow('Zrektyfikowany obraz prawy', rectified_img_R)
    cv.waitKey(0)

    (height, width) = rectified_img_L.shape[:2]
    for y in range(0, height, 50):
        cv.line(image_L, (0, y), (width, y), (255, 0, 0), 1)
        cv.line(image_R, (0, y), (width, y), (255, 0, 0), 1)
        cv.line(rectified_img_L, (0, y), (width, y), (255, 0, 0), 1)
        cv.line(rectified_img_R, (0, y), (width, y), (255, 0, 0), 1)
        


    plt.subplot(221),plt.imshow(image_L)
    plt.subplot(222),plt.imshow(image_R)
    plt.subplot(223),plt.imshow(rectified_img_L)
    plt.subplot(224),plt.imshow(rectified_img_R)
    plt.show()

    # zad 6
    cv.imwrite("retf_undistorted_L.png", rectified_img_L)