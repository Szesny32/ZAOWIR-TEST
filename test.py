import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from time import time
from copy import copy

CHECKBOARD_SIZE = (8, 11)
CHECKBOARD_PATTERN = (7, 10)
CHECKBOARD_NO_CORNERS = 70
SQUARE_SIZE = 44
MARKER_SIZE = 34
UNIT = 'mm'

CHARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
CHARUCO_BOARD = cv.aruco.CharucoBoard(CHECKBOARD_SIZE, SQUARE_SIZE, MARKER_SIZE, CHARUCO_DICT)
DETECTOR_PARAMETERS = cv.aruco.DetectorParameters()
CHARUCO_PARAMS = cv.aruco.CharucoParameters()
CHARUCO_DETECTOR = cv.aruco.CharucoDetector(CHARUCO_BOARD, CHARUCO_PARAMS, DETECTOR_PARAMETERS)

DEBUG = False


LEFT_PATH = 'datasets/Charuco3/'
RIGHT_PATH = 'datasets/Charuco3/'

RED = (0, 0, 255)
GREEN = (0, 255, 0)
objPointsArray = []
#------------------------------------------------------------------------------------------


def LoadDataset(path, prefix, extension):
    dataset = [ path+file for file in os.listdir(path) if ((prefix == None or file.startswith(prefix)) and (extension == None or file.endswith(extension))) ]
    print(f"Znaleziono {len(dataset)} próbek oznaczonych: '{prefix}' z rozszerzeniem: '{extension}'")
    assert len(dataset) > 0
    return dataset

def initImageSize(sample):
    image = cv.imread(sample)
    height, width = image.shape[:2]
    return (width, height)

def Calibration(_DATASET):
    imageSize = initImageSize(_DATASET[0])
    #Szukanie narożników
    imgPoints = []
    for sample in tqdm(_DATASET):
        image = cv.imread(sample)

        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image, CHARUCO_DICT)
        response, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(corners, ids, image, CHARUCO_BOARD)
        #charuco_corners, charuco_ids, marker_corners, marker_ids = CHARUCO_DETECTOR.detectBoard(image)

        if len(charuco_corners) == CHECKBOARD_NO_CORNERS:
            #_, imgPts = cv.aruco.getBoardObjectAndImagePoints(CHARUCO_BOARD, charuco_corners, charuco_ids)
            imgPoints.append(charuco_corners)

        if DEBUG == True:
            image = cv.aruco.drawDetectedCornersCharuco(image, charuco_corners, charucoIds=None, cornerColor=(0,0,255))
            image = ScaleImage(image, 1000)
            cv.imshow(f"Charuco board: {sample}", image)
            cv.waitKey(2500)
            cv.destroyAllWindows()
    
    assert len(imgPoints) > 0
    print(f'Nadaje się {len(imgPoints)}/{len(_DATASET)} próbek.')


    #Kalibracja
    timestamp = time()
    print(f"Rozpoczęto proces kalibracji dla {len(imgPoints)} próbek.")

    objPoints = GenerateObjectPoints()
    objPointsArray = [objPoints for _ in range(len(imgPoints))]  
     



    retval, cameraMatrix, distCoeffs, rvecs, tvecs =  cv.calibrateCamera(objPointsArray, imgPoints, imageSize, cameraMatrix= None, distCoeffs= None) 
    print(f"Zakończono proces kalibracji w {(time() - timestamp):.2f}s\n")
    return (retval, cameraMatrix, distCoeffs, rvecs, tvecs, imgPoints) 
    

def ScaleImage(image, base = 600):
    height, width = image.shape[:2]
    scale = base / height
    (height, width) = (int(scale * height), int(scale * width))
    return cv.resize(image, (width, height))

def GenerateObjectPoints():

    objp = np.zeros((CHECKBOARD_PATTERN[0] * CHECKBOARD_PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKBOARD_PATTERN[0], 0 : CHECKBOARD_PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE
    return objp

def CalculateFOV(cameraMatrix, imageSize):

    (width, height)  = imageSize

    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]

    fovW = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fovH = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi

    return fovW, fovH

def StereoCalibration(calibration1, calibration2, imageSize):
    (_, cameraMatrix1, distCoeffs1, _, _, imgPoints1) = calibration1
    (_, cameraMatrix2, distCoeffs2, _, _, imgPoints2) = calibration2

    objPoints = GenerateObjectPoints()
    objPointsArray = [objPoints for j in range(len(imgPoints1))]   

    stereoData = cv.stereoCalibrate(objPointsArray, imgPoints1, imgPoints2, 
                        cameraMatrix1, distCoeffs1,
                        cameraMatrix2, distCoeffs2, 
                        imageSize)
    return stereoData

def ExportCalibration(filepath, data):
    (retval,  cameraMatrix,  distCoeffs,  rvecs,  tvecs, _) = data
    data = {
        "retval" : retval,
        "cameraMatrix" : cameraMatrix.tolist(), 
        "distCoeffs" : distCoeffs.tolist(),
        "rvecs" : [rvec.tolist() for rvec in rvecs],
        "tvecs" : [tvec.tolist() for tvec in tvecs]
    }
    with open(filepath, 'w') as file:
        json.dump(data, file, separators=(',', ': '), indent = 4)
    print(f"Konfiguracja kalibracyjna została pomyślnie zapisana w pliku {filepath}\n")

def ExportStereo(path, data):
    (retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) =  data
    jsonData = {
        "retval" : retval,
        "cameraMatrix1" : cameraMatrix1.tolist(), 
        "distCoeffs1" : distCoeffs1.tolist(),
        "cameraMatrix2" : cameraMatrix2.tolist(), 
        "distCoeffs2" : distCoeffs2.tolist(),
        "rotationMat" : rotationMat.tolist(),
        "translationVec" : translationVec.tolist(),
        "essentialMat" : essentialMat.tolist(),
        "fundamentalMat" : fundamentalMat.tolist()
    }
    with open(path, 'w') as file:
        json.dump(jsonData, file, separators=(',', ': '), indent = 4)
        print(f"Konfiguracja kalibracyjna została pomyślnie zapisana w pliku {path}\n")

def Rectify(image, cameraMatrix, distCoeffs, R1, P1, sroi, IMAGE_SIZE):
    map1, map2 = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, R1, P1, IMAGE_SIZE, m1type=cv.CV_32FC1)
    rect = cv.remap(image, map1, map2, interpolation=cv.INTER_LINEAR)
    cv.rectangle(rect, (sroi[0], sroi[1]), (sroi[2], sroi[3]), color=GREEN, thickness=2)
    return rect

#----------------------------------------------
#[...] Wczytatanie danych
LEFT_DATASET = LoadDataset(LEFT_PATH, prefix='left', extension='.png')
RIGHT_DATASET = LoadDataset(RIGHT_PATH, prefix='right', extension='.png')
IMAGE_SIZE = initImageSize(LEFT_DATASET[0])


#[#..] Kalibracja
LEFT_CALIBRATION = Calibration(LEFT_DATASET)
ExportCalibration('LEFT_CALIBRATION.json', data = LEFT_CALIBRATION)

RIGHT_CALIBRATION = Calibration(RIGHT_DATASET)
ExportCalibration('RIGHT_CALIBRATION.json', data = RIGHT_CALIBRATION)


#[##.] Stereo
stereoData = StereoCalibration(LEFT_CALIBRATION, RIGHT_CALIBRATION, IMAGE_SIZE)
(retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) = stereoData
ExportStereo('STEREO_CALIBRATION.json', stereoData)

baseline_mm = np.linalg.norm(translationVec)
baseline_cm = baseline_mm * 0.1
print(f"Odległość bazowa (baseline): {baseline_cm:.2f} cm")
print(f'LEFT FOV: {CalculateFOV(cameraMatrix1, IMAGE_SIZE)}')
print(f'RIGHT FOV: {CalculateFOV(cameraMatrix2, IMAGE_SIZE)}')


#[###] Rektyfikacja
img1 = cv.imread(LEFT_DATASET[0])
img2 = cv.imread(RIGHT_DATASET[0])

R1, R2, P1, P2, Q, sroi1, sroi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, IMAGE_SIZE, R=rotationMat, T=translationVec, alpha=1)

rect1 = Rectify(img1, cameraMatrix1, distCoeffs1, R1, P1, sroi1, IMAGE_SIZE)
rect2 = Rectify(img2, cameraMatrix2, distCoeffs2, R2, P2, sroi2, IMAGE_SIZE)
outputImg = np.concatenate((rect1, rect2), axis=1)
height, width = outputImg.shape[:2]

for i in range(10, height, 50):
    cv.line(outputImg, (0, i), (width, i), color=RED,  thickness=2)
cv.imwrite("outputImg.png", outputImg)



