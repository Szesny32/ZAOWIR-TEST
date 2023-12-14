import Tools
import Calibration
import cv2 as cv

CHECKBOARD_PATTERN = (8, 6)
WINDOW_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
path = "datasets/Lab1/s1/"
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
}

CALIBRATION_SAMPLES = 80
CHECKBOARD_SIZE = 2.867
example = "datasets/Lab1/s1/left_0.png"
    

def task1():
    dataset_L = Tools.LoadDataset(path, prefix= prefix_L, extension= extension)
    corners_dataset_L = Calibration.FindCorners(dataset_L, pattern= CHECKBOARD_PATTERN, export_path= WORKSPACE['corners_L'], downscale= False)
    subcorners_dataset_L = Calibration.FindSubPixelCorners(corners_dataset_L, subpixel_params=(WINDOW_SIZE, ZERO_ZONE, CRITERIA), export_path= WORKSPACE['subcorners_L'], downscale= False)

    dataset_R = Tools.LoadDataset(path, prefix= prefix_R, extension= extension)
    corners_dataset_R = Calibration.FindCorners(dataset_R, pattern= CHECKBOARD_PATTERN,  export_path=WORKSPACE['corners_R'], downscale= False)
    subcorners_dataset_R = Calibration.FindSubPixelCorners(corners_dataset_R, subpixel_params=(WINDOW_SIZE, ZERO_ZONE, CRITERIA),export_path= WORKSPACE['subcorners_R'], downscale= False)
    task1_result(show = True)


def task1_result(show = False):
    corners_dataset_L = Tools.ImportCorners(WORKSPACE['corners_L'])
    corners_dataset_R = Tools.ImportCorners(WORKSPACE['corners_R'])
    subcorners_dataset_L = Tools.ImportCorners(WORKSPACE['subcorners_L'])
    subcorners_dataset_R = Tools.ImportCorners(WORKSPACE['subcorners_R'])

    if show:
        print(f'\nL: {Tools.ListCornersIds(corners_dataset_L, len(path+prefix_L), len(extension))}\n') 
        Tools.DisplayImagesWithChessoardCorners(corners_dataset_L, subcorners_dataset_L, pattern= CHECKBOARD_PATTERN)
        print(f'\nR: {Tools.ListCornersIds(corners_dataset_R, len(path+prefix_R), len(extension))}\n')
        Tools.DisplayImagesWithChessoardCorners(corners_dataset_R, subcorners_dataset_R, pattern= CHECKBOARD_PATTERN)

    return (corners_dataset_L, corners_dataset_R, subcorners_dataset_L, subcorners_dataset_R)


def task2():

    (corners_dataset_L, corners_dataset_R, subcorners_dataset_L, subcorners_dataset_R) = task1_result()

    corners_L = [corners for _, corners in corners_dataset_L][:CALIBRATION_SAMPLES]
    corners_R = [corners for _, corners in corners_dataset_R][:CALIBRATION_SAMPLES]
    sub_corners_L = [corners for _, corners in subcorners_dataset_L][:CALIBRATION_SAMPLES]
    sub_corners_R = [corners for _, corners in subcorners_dataset_R][:CALIBRATION_SAMPLES]

    corners_calibration_L = Calibration.Calibrate(imgPoints = corners_L, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , example = example)
    Tools.ExportCalibration(WORKSPACE['calibration_L'], corners_calibration_L)

    sub_corners_calibration_L = Calibration.Calibrate(imgPoints = sub_corners_L, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , example = example)
    Tools.ExportCalibration(WORKSPACE['sub_calibration_L'], sub_corners_calibration_L)

    corners_calibration_R = Calibration.Calibrate(imgPoints = corners_R, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , example = example)
    Tools.ExportCalibration(WORKSPACE['calibration_R'], corners_calibration_R)

    sub_corners_calibration_R = Calibration.Calibrate(imgPoints = sub_corners_R, pattern = CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE , example = example)
    Tools.ExportCalibration(WORKSPACE['sub_calibration_R'], sub_corners_calibration_R)


def task2_result():
    calibration_L = Tools.ImportCalibration(WORKSPACE['calibration_L'])
    sub_calibration_L = Tools.ImportCalibration(WORKSPACE['sub_calibration_L'])
    calibration_R = Tools.ImportCalibration(WORKSPACE['calibration_R'])
    sub_calibration_R = Tools.ImportCalibration(WORKSPACE['sub_calibration_R'])
    return (calibration_L, calibration_R, sub_calibration_L, sub_calibration_R)

def task3():
    (corners_dataset_L, corners_dataset_R, subcorners_dataset_L, subcorners_dataset_R) = task1_result()
    (calibration_L, calibration_R, sub_calibration_L, sub_calibration_R) = task2_result()

    corners_L = [corners for _, corners in corners_dataset_L][:CALIBRATION_SAMPLES]
    sub_corners_L = [corners for _, corners in subcorners_dataset_L][:CALIBRATION_SAMPLES]
    
    Calibration.MeanReprojectionError(calibration_L, corners_L, pattern=CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE)
    print("Subpikselowo: ")
    Calibration.MeanReprojectionError(sub_calibration_L, sub_corners_L, pattern=CHECKBOARD_PATTERN, side_length = CHECKBOARD_SIZE)


def task5():
    (calibration_L, _) = task2_result()
    Calibration.Undisort(WORKSPACE['disortImage'], WORKSPACE['undisortImage'], calibration_L)
    Tools.CompareImages(WORKSPACE['disortImage'], "Disorted Image", WORKSPACE['undisortImage'], "Undisorted Image")
    
def task6():
    (calibration_L, _) = task2_result()
    Calibration.InitUndisortRectifyMap(WORKSPACE['disortImage'], WORKSPACE['undisortImage'], calibration_L)
    Tools.CompareImages(WORKSPACE['disortImage'], "Disorted Image", WORKSPACE['undisortImage'], "Undisorted Image")
    
