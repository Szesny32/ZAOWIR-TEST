import Tools
import Stereo
import Calibration
import cv2 as cv
import numpy as np

from copy import copy
#pip install opencv-contrib-python
PATH = 'datasets/Charuco3/'
EXTENSION = '.png'
DATASET = Tools.LoadDataset(PATH, prefix=None, extension=EXTENSION)
CHARUCO_BOARD_PATTERN = (8, 11)
CHESS_BOARD_PATTERN = (CHARUCO_BOARD_PATTERN[0]-1, CHARUCO_BOARD_PATTERN[1]-1)
CHARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
CHARUCO_SQUARE_LENGTH = 44
CHARUCO_MARKER_LENGTH = 34
CHARUCO_BOARD= cv.aruco.CharucoBoard(CHARUCO_BOARD_PATTERN, CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, CHARUCO_DICT)
DEBUG = False
#(_corners, _ids) = ([], [])

# for sample in DATASET:
#     image = cv.imread(sample)
#     height, width = image.shape[:2]

#     corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image, CHARUCO_DICT)
#     response, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(corners, ids, image, CHARUCO_BOARD)
#     print(charuco_ids)

#     if response == (CHARUCO_BOARD_PATTERN[0]-1) * (CHARUCO_BOARD_PATTERN[1]-1):
#         _corners.append(charuco_corners)
#         _ids.append(charuco_ids)

#         if DEBUG == True:
#             image = cv.aruco.drawDetectedCornersCharuco(image, charuco_corners, charucoIds=None, cornerColor=(0,0,255))
#             print(sample)
#             cv.imshow(f"Charuco board: {sample}", image)
#             cv.waitKey(2500)
#             cv.destroyAllWindows()

# print(f'{len(_corners)}/{len(DATASET)}')

# calibration = Calibration.Calibrate(_corners, CHESS_BOARD_PATTERN, CHARUCO_SQUARE_LENGTH, imageSize= (width, height))
# (retval, cameraMatrix, distCoeffs, rvec, tvecs) = calibration



L = ['datasets/Charuco3/left.png']
image = cv.imread(L[0])
corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image, CHARUCO_DICT)
response, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(corners, ids, image, CHARUCO_BOARD)
L_CORNERS  = [charuco_corners]



R = ['datasets/Charuco3/right.png']
image = cv.imread(R[0])
corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(image, CHARUCO_DICT)
response, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(corners, ids, image, CHARUCO_BOARD)
R_CORNERS  = [charuco_corners]
height, width = image.shape[:2]



calibration1 = Calibration.Calibrate(L_CORNERS, CHESS_BOARD_PATTERN, CHARUCO_SQUARE_LENGTH, imageSize= (width, height))
calibration2 = Calibration.Calibrate(R_CORNERS, CHESS_BOARD_PATTERN, CHARUCO_SQUARE_LENGTH, imageSize= (width, height))



stereoData = Stereo.Calibrate(L_CORNERS, R_CORNERS, calibration1, calibration2, pattern = CHESS_BOARD_PATTERN, side_length = CHARUCO_SQUARE_LENGTH , imageSize= (width, height))
(retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, rotationMat, translationVec, essentialMat, fundamentalMat) =  stereoData
baseline = np.linalg.norm(translationVec) 
print(f"Odległość bazowa (baseline): {baseline:.2f} cm")

image = Tools.ScaleImage(image, base = 1000)
height, width = image.shape[:2]
(fovW, fovH) = Tools.Fov(cameraMatrix1,  imgSize=(width, height))
print(f'FOV: {(fovW, fovH)}')

import math
fov_w_rad = math.radians(fovW)
fov_h_rad = math.radians(fovH)

# Obliczenia FOV przekątnej
fov_diag_rad = math.sqrt(fov_w_rad**2 + fov_h_rad**2)

# Konwersja z powrotem na stopnie
fov_diag_deg = math.degrees(fov_diag_rad)
print(f"FOV przekątnej: {fov_diag_deg:.2f} stopni")



