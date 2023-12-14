from tqdm import tqdm
from time import time
from copy import copy
import cv2 as cv
import Tools

def FindCorners(dataset, pattern, export_path, downscale = False):
    timestamp = time()
    (corners_dataset, unknown) = ([], [])
    for filepath in tqdm(dataset):
        try:
            image = cv.imread(filepath)
            if(downscale == True):
                image = Tools.Downscale(image)

            ret, corners = cv.findChessboardCorners(image, pattern) 
            if ret:
                corners_dataset.append((filepath, corners))
            else: 
                unknown.append(filepath)
        except Exception as e:
            unknown.append(filepath)
            print(f"\nWystąpił błąd podczas odczytywania obrazu - {filepath} : {e}\n")

    print(f"\nZakończono proces wykrywania wzorca: {(time() - timestamp):.2f}s")
    print(f"Wzorzec znaleziono w {len(corners_dataset)}/{len(dataset)} próbkach")
    print(f"Wzorca nie znaleziono w {len(unknown)}/{len(dataset)} próbkach")
    Tools.ExportCorners(corners_dataset, export_path)
    return corners_dataset             
            
def FindSubPixelCorners(corners_dataset, subpixel_params, export_path,  downscale = False):
    (winSize, zeroZone, criteria) = subpixel_params

    timestamp = time()
    subcorners_dataset = []
    print(f"Rozpoczęto proces subpikselowego wykrywania wzorca.")
    for filepath, corners in tqdm(corners_dataset):
        try:
            image = cv.imread(filepath)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            if(downscale == True):
                image = Tools.ScaleImage(image)
            subPixCorners = cv.cornerSubPix(image, copy(corners), winSize, zeroZone, criteria)
            subcorners_dataset.append((filepath, subPixCorners))
        except Exception as e:
            print(f"\n\nWystąpił błąd podczas odczytywania obrazu - {filepath} : {e}\n\n")
    print(f"\nZakończono proces subpikselowego wykrywania wzorca w {(time() - timestamp):.2f}s\n")
    Tools.ExportCorners(subcorners_dataset, export_path)
    return subcorners_dataset
              

def Calibrate(imgPoints, pattern, side_length, imageSize):
    timestamp = time()
    print(f"Rozpoczęto proces kalibracji dla {len(imgPoints)} próbek")
    objPoints = Tools.GenerateObjectPoints(pattern, side_length)
    objPointsArray = [objPoints for _ in range(len(imgPoints))]   
    print((objPointsArray, imgPoints, imageSize))
    retval, cameraMatrix, distCoeffs, rvecs, tvecs =  cv.calibrateCamera(objPointsArray, imgPoints, imageSize, cameraMatrix= None, distCoeffs= None) 
 
    print(f"Zakończono proces kalibracji w {(time() - timestamp):.2f}s\n")
    return (retval,  cameraMatrix,  distCoeffs,  rvecs,  tvecs)

   
def MeanReprojectionError(calibrationData, imgPoints, pattern, side_length):
    (retval, cameraMatrix, distCoeffs, rvecs, tvecs) = calibrationData
    print(f"Rozpoczęto proces wyznaczanie średniego błędu reprojekcji dla {len(imgPoints)} próbek")
    n = len(imgPoints)
    objPoints = Tools.GenerateObjectPoints(pattern, side_length)
    objPointsArray = [objPoints for i in range(n)]  
    
    timestamp = time()
    meanError = 0
    for i in range(n):
        projectPoints, _ = cv.projectPoints(objPointsArray[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        error = cv.norm(imgPoints[i], projectPoints, cv.NORM_L2) / len(projectPoints)
        meanError += error
    meanError/=n
    print(f"\nZakończono proces wyznaczanie średniego błędu reprojekcji w {(time() - timestamp):.2f}s\n")
    print(f"Średni błąd reprojekcji: {meanError:.4f}")
    return meanError
#-------------------------

def Undisort(disortImage, undisortImage, calibrationData):
    print( f"\nUsuwanie dystorsji na obrazie - metoda undisort\n")
    img = cv.imread(disortImage)
    height, width = img.shape[:2]
    (retval, cameraMatrix, distCoeffs, rvecs, tvecs) = calibrationData

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (width, height), 1, (width, height))
    dst = cv.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(undisortImage, dst)


def InitUndisortRectifyMap(disortImage, undisortImage, calibrationData):
    print( f"\nUsuwanie dystorsji na obrazie - metoda initUndisortRectifyMap i remap\n")
    img = cv.imread(disortImage)
    height, width = img.shape[:2]
    (retval, cameraMatrix, distCoeffs, rvecs, tvecs) = calibrationData
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (width, height), 1,(width, height))

    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newcameramtx, (width, height), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(undisortImage, dst)   