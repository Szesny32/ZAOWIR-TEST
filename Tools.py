import os
import sys
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def LoadDataset(path, prefix, extension):
    dataset = [ path+file for file in os.listdir(path) if ((prefix == None or file.startswith(prefix)) and (extension == None or file.endswith(extension))) ]
    print(f"Znaleziono {len(dataset)} próbek oznaczonych: '{prefix}' z rozszerzeniem: '{extension}'")
    assert len(dataset) > 0
    return dataset

def GetImageSize(image):
    return cv.imread(image).shape[:2]

def Downscale(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return ScaleImage(gray) 

def ScaleImage(image, base = 600):
    height, width = image.shape[:2]
    scale = base / height
    (height, width) = (int(scale * height), int(scale * width))
    return cv.resize(image, (width, height))

def DisplayImagesWithChessoardCorners(corners_data, subcorners_data, pattern):
    dist = 0
    best_i = 0
    for i in range(len(corners_data)):
        _, corners = corners_data[i]
        _, subcorners = subcorners_data[i]

        d = np.sum((corners - subcorners)**2)
        if d > dist:
            dist = d
            best_i = i

    filepath1, corners = corners_data[best_i]
    filepath2, subcorners = subcorners_data[best_i]

    image1 = cv.imread(filepath1)
    image1 = cv.drawChessboardCorners(image1, pattern, corners, patternWasFound=True)
    image1 = ScaleImage(image1)

    image2 = cv.imread(filepath2)
    image2 = cv.drawChessboardCorners(image2, pattern, subcorners, patternWasFound=True)
    image2 = ScaleImage(image2)

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    plt.title(f'Corners: {filepath1}', fontsize= 20)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB)) 
    plt.title(f'SubCorners: {filepath2}', fontsize= 20)
    plt.axis('off')

    plt.show()


def ExportCorners(data, filepath):
    serializableData = [(filepath, corners.tolist()) for filepath, corners in data]
    with open(filepath, 'w') as json_file:
        json.dump(serializableData, json_file) 

def ImportCorners(filepath):
    with open(filepath, 'r') as json_file:
        loaded_data = json.load(json_file)
    data = []
    for item in loaded_data:
        filepath, corners_list = item
        corners = np.array(corners_list, dtype=np.float32)
        data.append((filepath, corners))
    return data

def ListCornersIds(data, start, end):
    ids = []
    for path, _ in data:
        ids.append(path[start: -end])
    return ids

def Wait():
    while True:
        key = cv.waitKey(0)
        if key == 13 or key == 27: 
            cv.destroyAllWindows() 
            return key  

def Fov(cameraMatrix, imgSize):
    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]
    width, height = imgSize

    fovW = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
    fovH = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi

    return fovW, fovH


def GenerateObjectPoints(pattern_size, side_length):
    x = pattern_size[0]
    y = pattern_size[1]
    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : x, 0 : y].T.reshape(-1, 2) * side_length
    return objp

def ListCalibrationResults(calibrationData):
    (retval,  cameraMatrix,  distCoeffs,  rvecs,  tvecs) = calibrationData
    print("\nUzyskane wyniki kalibracji:")
    print(f"retval: {retval}")
    print(f"cameraMatrix: {cameraMatrix}")
    print(f"distCoeffs: {distCoeffs}")
    print(f"rvecs: {rvecs}")
    print(f"tvecs: {tvecs}")


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


def ImportCalibration(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    calibrationData = (data['retval'],  np.array(data["cameraMatrix"]),  np.array(data["distCoeffs"]),  [np.array(rvec) for rvec in data["rvecs"]], [np.array(tvec) for tvec in data["tvecs"]])
    print(f"Konfiguracja kalibracyjna została pomyślnie wczytana z pliku {filepath}\n")
    return calibrationData  

def CompareImages(_img1, _desc1, _img2, _desc2):
    img1 = cv.imread(_img1)
    img2 = cv.imread(_img2)
    plt.subplot(1, 2, 1)
    plt.imshow(ScaleImage(img1))
    plt.title(f'{_desc1}', fontsize= 20)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ScaleImage(img2)) 
    plt.title(f'{_desc2}', fontsize= 20)
    plt.axis('off')
    plt.show()


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

def ImportStereo(path):
    with open(path, 'r') as file:
        data = json.load(file)

    stereoData = (
        data['retval'],
        np.array(data['cameraMatrix1']), 
        np.array(data['distCoeffs1']),
        np.array(data['cameraMatrix2']), 
        np.array(data['distCoeffs2']),
        np.array(data['rotationMat']), 
        np.array(data['translationVec']),
        np.array(data['essentialMat']), 
        np.array(data['fundamentalMat']),
        )
    
    print(f"Konfiguracja kalibracyjna została pomyślnie wczytana z pliku {path}\n")
    return stereoData