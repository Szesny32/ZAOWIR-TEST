import Tools
import cv2 as cv

def AssociateFrames(dataset_L, dataset_R, dataset_metadata, number_of_samples = 30):

    (path, prefix_L, prefix_R, extension) = dataset_metadata
    (imgPoints_L, imgPoints_R, i) = ([], [], 0)
    n = min(len(dataset_L), len(dataset_R))
    print(f'\nPatterns were detected in the associated images for id:')
    
    ids_L = Tools.ListCornersIds(dataset_L, len(path+prefix_L), len(extension))
    ids_R = Tools.ListCornersIds(dataset_R, len(path+prefix_R), len(extension))
    associate_ids = []
    for id in range(n):
        
        if i >= number_of_samples:
            break
        image_L = f'{path}{prefix_L}{id}{extension}'
        image_R = f'{path}{prefix_R}{id}{extension}'
        
        if  f'{id}' in ids_L and f'{id}' in ids_R:
            i += 1
            associate_ids.append(id)
            imgPoints_L.append(next(((path, corner) for path, corner in dataset_L if path == image_L), None))
            imgPoints_R.append(next(((path, corner) for path, corner in dataset_R if path == image_R), None))
    print(associate_ids)
    return (imgPoints_L, imgPoints_R)


def Calibrate(imgPoints1, imgPoints2, calibration_L, calibration_R, pattern, side_length, imageSize):
    (_, cameraMatrix1, distCoeffs1, _, _) = calibration_L
    (_, cameraMatrix2, distCoeffs2, _, _) = calibration_R

    objPoints = Tools.GenerateObjectPoints(pattern, side_length)
    objPointsArray = [objPoints for j in range(len(imgPoints1))]   

    stereoData = cv.stereoCalibrate(objPointsArray, imgPoints1, imgPoints2, 
                        cameraMatrix1, distCoeffs1,
                        cameraMatrix2, distCoeffs2, 
                        imageSize)
    return stereoData