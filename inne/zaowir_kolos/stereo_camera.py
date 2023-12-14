import os
import sys
import argparse
import json
import time
from json import JSONEncoder
import glob
import cv2 as cv
import numpy as np

CALIB_CAM1_PATH = "stereo_calibration_cam1.json"
CALIB_CAM2_PATH = "stereo_calibration_cam2.json"

STEREO_CALIB_PATH = "calibration_stereo.json"
ALPHA_SCALE = 0
SINGLE_SQUARE = 44
CHESSBOARD = (7,10)
square_length_mm = 44.0
marker_length_mm = 34.0
ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
IMAGE_CAM1_PATH = "left.png"
IMAGE_CAM2_PATH = "right.png"
OUTPUT_PATH = 'rectified.png'

def calculate_fov(smtx, imgSize):
    fx = smtx[0][0]
    fy = smtx[1][1]
    width, height = imgSize

    fovW = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fovH = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi

    return fovW, fovH

def stereo_calibrate(objPoints, imgPoints1, imgPoints2, imgSize, mtx1, mtx2, dist1, dist2):
    # Read stereo calibration from file
    flags = cv.CALIB_FIX_INTRINSIC
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    retval, smtx1, sdist1, smtx2, sdist2, R, T, E, F = cv.stereoCalibrate(objPoints, imgPoints1, imgPoints2, mtx1, dist1, mtx2, dist2, imgSize, criteria=criteria, flags=flags)    
    baseline = round(np.linalg.norm(T)*0.1, 2)
    fov_cam1 = calculate_fov(smtx1, image_size1) 
    fov_cam2 = calculate_fov(smtx2, image_size2)

    print(f'Baseline: {baseline}cm')
    print(f'FOV of cam1: {fov_cam1}')
    print(f'FOV of cam2: {fov_cam2}')
    return (smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2)


def rectification(smtx1, dist1, smtx2, dist2, imgSize, R, T):

    rectify_scale = ALPHA_SCALE
    R1, R2, P1, P2, Q, sroi1, sroi2 = cv.stereoRectify(smtx1, dist1, smtx2, dist2, imgSize, R, T, alpha=rectify_scale)
    map1_cam1, map2_cam1 = cv.initUndistortRectifyMap(smtx1, dist1, R1, P1, imgSize, m1type=cv.CV_16SC2)
    map1_cam2, map2_cam2 = cv.initUndistortRectifyMap(smtx2, dist2, R2, P2, imgSize, m1type=cv.CV_16SC2)
    return map1_cam1, map2_cam1, map1_cam2, map2_cam2, sroi1, sroi2


def remapAndShowRect(map1_cam1, map2_cam1, map1_cam2, map2_cam2, image_cam1, image_cam2, sroi1, sroi2):
    
    img1 = cv.imread(image_cam1)
    img2 = cv.imread(image_cam2)

    remap_method = cv.INTER_LINEAR
    imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, remap_method)
    imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, remap_method)    

    cv.rectangle(imgAfterRect1, (sroi1[0], sroi1[1]), (sroi1[2], sroi1[3]), (0, 255, 0), 2)
    cv.rectangle(imgAfterRect2, (sroi2[0], sroi2[1]), (sroi2[2], sroi2[3]), (0, 255, 0), 2)

    catImgsAfterRect = np.concatenate((imgAfterRect1, imgAfterRect2), axis=1)
    for i in range(10, np.shape(catImgsAfterRect)[0], 50):
       cv.line(catImgsAfterRect, (0, i), (np.shape(catImgsAfterRect)[1], i), (0, 0, 0), 1)

    cv.imwrite(OUTPUT_PATH, catImgsAfterRect)


# Save final parameter matrix to json
def save_stereo_to_json(smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2):
    # >>> Saving result to json <<<
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    json_data = {
        'smtx1': smtx1,
        'sdist1': sdist1,
        'smtx2': smtx2,
        'sdist2': sdist2,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'Baseline': baseline,
        'Fov_cam1': fov_cam1,
        'Fov_cam2': fov_cam2
    }

    # Writing to json
    with open(STEREO_CALIB_PATH, "w") as outfile:
        json.dump(json_data, outfile, indent=4, cls=NumpyArrayEncoder)

def calibration_procedure():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)                                 # Define [70][3] array filled with zeros
    objp[:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SINGLE_SQUARE        # Fills indexies to elements

    # Arrays to store object points and image points from all the images
    objpoints_cam1 = [] # 3d point in real world space
    imgpoints_cam1 = [] # 2d points in image plane [Corner sub pix]
    objpoints_cam2 = [] # 3d point in real world space
    imgpoints_cam2 = [] # 2d points in image plane [Corner sub pix]

    # Removing uncommon files

    img_cam1 = cv.imread(IMAGE_CAM1_PATH)
    img_cam2 = cv.imread(IMAGE_CAM2_PATH)

    gray_cam1 = cv.cvtColor(img_cam1, cv.COLOR_BGR2GRAY)
    gray_cam2 = cv.cvtColor(img_cam2, cv.COLOR_BGR2GRAY)

    ARUCO_PARAMETERS = cv.aruco.DetectorParameters()
    CHARUCO_PARAMS = cv.aruco.CharucoParameters()
    board = cv.aruco.CharucoBoard((CHESSBOARD[0]+1,CHESSBOARD[1]+1), square_length_mm, marker_length_mm, ARUCO_DICT)
    detector = cv.aruco.CharucoDetector(board, CHARUCO_PARAMS, ARUCO_PARAMETERS)

    charucoCorners1, charucoIds1, markerCorners1, markerIds1 = detector.detectBoard(gray_cam1)
    charucoCorners2, charucoIds2, markerCorners2, markerIds2 = detector.detectBoard(gray_cam2)

    image_size_cam1 = gray_cam1.shape[::-1]
    image_size_cam2 = gray_cam2.shape[::-1]

    if image_size_cam1 != image_size_cam2:
        print('Couple rejected due to diffrent image sizes')

    # Only if both found

        # Refining pixel coordinates for given 2d points.
    cornersSubPix_cam1 = cv.cornerSubPix(gray_cam1, charucoCorners1, (11, 11), (-1, -1), criteria)
    cornersSubPix_cam2 = cv.cornerSubPix(gray_cam2, charucoCorners2, (11, 11), (-1, -1), criteria)

    objpoints_cam1.append(objp)
    objpoints_cam2.append(objp)

    imgpoints_cam1.append(cornersSubPix_cam1)
    imgpoints_cam2.append(cornersSubPix_cam2)

    mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1 = calibrate_camera(objpoints_cam1, imgpoints_cam1, image_size_cam1)

    mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2 = calibrate_camera(objpoints_cam2, imgpoints_cam2, image_size_cam2)

    return (mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1, image_size_cam1, mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2, image_size_cam2)

def save_to_json(mtx, dist, rvecs, tvecs, image_size, objpoints, imgpoints, output_path):
    # >>> Saving result to json <<<
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    json_result = {
        "mtx": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "imageSize": image_size,
        "objPoints": objpoints,
        "imgPoints": imgpoints
    }

    # Writing to json
    with open(output_path, "w") as outfile:
        json.dump(json_result, outfile, indent=4, cls=NumpyArrayEncoder)


# Calibrates camera based on input parametes
def calibrate_camera(objpoints, imgpoints, image_size):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # Object points conversion
    for i, objpoint in enumerate(objpoints):
        objpoints[i] = np.asarray(objpoint, dtype=np.float32)

    # Image points conversion
    for i, imgpoint in enumerate(imgpoints):
        imgpoints[i] = np.asarray(imgpoint, dtype=np.float32)

    return (mtx, dist, rvecs, tvecs, objpoints, imgpoints)

if __name__ == "__main__":
    # Console argument 

# Perform camera calibration
    (mtx1, dist1, rvecs1, tvecs1, objpoints1, imgpoints1, image_size1, 
    mtx2, dist2, rvecs2, tvecs2, objpoints2, imgpoints2, image_size2) = calibration_procedure()

    save_to_json(mtx1, dist1, rvecs1, tvecs1, image_size1, objpoints1, imgpoints1, CALIB_CAM1_PATH)
    save_to_json(mtx2, dist2, rvecs2, tvecs2, image_size2, objpoints2, imgpoints2, CALIB_CAM2_PATH)

    # Stereo calibration
    smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2 = stereo_calibrate(objpoints1, imgpoints1, imgpoints2, image_size1, mtx1, mtx2, dist1, dist2)
    save_stereo_to_json(smtx1, sdist1, smtx2, sdist2, R, T, E, F, baseline, fov_cam1, fov_cam2)

    map1_cam1, map2_cam1, map1_cam2, map2_cam2, sroi1, sroi2 = rectification(smtx1, sdist1, smtx2, sdist2, image_size1, R, T)
    remapAndShowRect(map1_cam1, map2_cam1, map1_cam2, map2_cam2, IMAGE_CAM1_PATH, IMAGE_CAM2_PATH, sroi1, sroi2)
