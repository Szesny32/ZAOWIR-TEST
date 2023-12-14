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


CHARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
CHARUCO_BOARD = cv.aruco.CharucoBoard(CHECKBOARD_SIZE, SQUARE_SIZE, MARKER_SIZE, CHARUCO_DICT)
DETECTOR_PARAMETERS = cv.aruco.DetectorParameters()
CHARUCO_PARAMS = cv.aruco.CharucoParameters()
CHARUCO_DETECTOR = cv.aruco.CharucoDetector(CHARUCO_BOARD, CHARUCO_PARAMS, DETECTOR_PARAMETERS)
DEBUG = False

#------------------------------------------------------------------------------------------



def LoadDataset(path, prefix, extension):
    dataset = [ path+file for file in os.listdir(path) if ((prefix == None or file.startswith(prefix)) and (extension == None or file.endswith(extension))) ]
    print(f"Znaleziono {len(dataset)} próbek oznaczonych: '{prefix}' z rozszerzeniem: '{extension}'")
    assert len(dataset) > 0
    return dataset


def Calibration(_DATASET):
    #Szukanie narożników
    imgPoints = []
    #_objPoints = []
    for sample in tqdm(_DATASET):
        image = cv.imread(sample)
        height, width = image.shape[:2]

        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image, CHARUCO_DICT)
        response, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(corners, ids, image, CHARUCO_BOARD)

        #charuco_corners, charuco_ids, marker_corners, marker_ids = CHARUCO_DETECTOR.detectBoard(image)
        
        

        if len(charuco_corners) == CHECKBOARD_NO_CORNERS:
            #_, imgPts = cv.aruco.getBoardObjectAndImagePoints(CHARUCO_BOARD, charuco_corners, charuco_ids)
            imgPoints.append(charuco_corners)
            #imgPoints.append(imgPts)
            #imgPoints_ids.append(charuco_ids)
            #_objPoints.append(objPoints)


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
     



    retval, cameraMatrix, distCoeffs, rvecs, tvecs =  cv.calibrateCamera(objPointsArray, imgPoints, imageSize =  (width, height), cameraMatrix= None, distCoeffs= None) 
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

def CalculateFOV(cameraMatrix, sample):

    image = cv.imread(sample)
    height, width = image.shape[:2]

    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]

    fovW = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fovH = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi

    return fovW, fovH

def StereoCalibration(calibration1, calibration2, sample):
    image = cv.imread(sample)
    height, width = image.shape[:2]
    (_, cameraMatrix1, distCoeffs1, _, _, imgPoints1) = calibration1
    (_, cameraMatrix2, distCoeffs2, _, _, imgPoints2) = calibration2

    objPoints = GenerateObjectPoints()
    objPointsArray = [objPoints for j in range(len(imgPoints1))]   

    stereoData = cv.stereoCalibrate(objPointsArray, imgPoints1, imgPoints2, 
                        cameraMatrix1, distCoeffs1,
                        cameraMatrix2, distCoeffs2, 
                        imageSize = (width, height))
    return stereoData

def ExportCalibration(filepath, data):
    (retval,  cameraMatrix,  distCoeffs,  rvecs,  tvecs) = data
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


#----------------------------------------------
LEFT_PATH = 'datasets/Charuco3/'
LEFT_DATASET = LoadDataset(LEFT_PATH, prefix='left', extension='.png')

RIGHT_PATH = 'datasets/Charuco3/'
RIGHT_DATASET = LoadDataset(RIGHT_PATH, prefix='right', extension='.png')
#-
LEFT_CALIBRATION = Calibration(LEFT_DATASET)
(retval, cameraMatrix, distCoeffs, rvecs, tvecs, imgPoints)  = LEFT_CALIBRATION
ExportCalibration('LEFT_CALIBRATION.json', data = (retval, cameraMatrix, distCoeffs, rvecs, tvecs))


RIGHT_CALIBRATION = Calibration(RIGHT_DATASET)
(retval, cameraMatrix, distCoeffs, rvecs, tvecs, imgPoints)  = RIGHT_CALIBRATION
ExportCalibration('RIGHT_CALIBRATION.json', data = (retval, cameraMatrix, distCoeffs, rvecs, tvecs))



stereoData = StereoCalibration(LEFT_CALIBRATION, RIGHT_CALIBRATION, LEFT_DATASET[0])
(retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) =  stereoData
ExportStereo('STEREO_CALIBRATION.json', stereoData)


baseline = round(np.linalg.norm(translationVec), 2)
print(f"Odległość bazowa (baseline): {baseline/10:.2f} CM")

print(f'LEFT FOV: {CalculateFOV(cameraMatrix1, LEFT_DATASET[0])}')
print(f'RIGHT FOV: {CalculateFOV(cameraMatrix2, RIGHT_DATASET[0])}')


img1 = cv.imread(LEFT_DATASET[0])
img2 = cv.imread(RIGHT_DATASET[0])

(height, width) = img1.shape[:2]



R1, R2, P1, P2, Q, sroi1, sroi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (width, height), rotationMat, translationVec, alpha=0.9)
map1_cam1, map2_cam1 = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), m1type=cv.CV_32FC1)
map1_cam2, map2_cam2 = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), m1type=cv.CV_32FC1)





remap_method = cv.INTER_LINEAR
imgAfterRect1 = cv.remap(img1, map1_cam1, map2_cam1, remap_method)
imgAfterRect2 = cv.remap(img2, map1_cam2, map2_cam2, remap_method)    
cv.rectangle(imgAfterRect1, (sroi1[0], sroi1[1]), (sroi1[2], sroi1[3]), (0, 0, 255), 2)
cv.rectangle(imgAfterRect2, (sroi2[0], sroi2[1]), (sroi2[2], sroi2[3]), (0, 0, 255), 2)



catImgsAfterRect = np.concatenate((imgAfterRect1, imgAfterRect2), axis=1)
for i in range(10, np.shape(catImgsAfterRect)[0], 50):
    cv.line(catImgsAfterRect, (0, i), (np.shape(catImgsAfterRect)[1], i), (0, 0, 255), 1)

cv.imwrite("catImgsAfterRect.png", catImgsAfterRect)



