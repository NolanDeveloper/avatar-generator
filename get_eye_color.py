import argparse as ag
import cv2
import dlib
from collections import OrderedDict
import numpy as np


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0] : ((166, 21, 50), (240, 100, 85)),
    class_name[1] : ((166, 2, 25), (300, 20, 75)),
    class_name[2] : ((2, 20, 20), (40, 100, 60)),
    class_name[3] : ((20, 3, 30), (65, 60, 60)),
    class_name[4] : ((0, 10, 5), (40, 40, 25)),
    class_name[5] : ((60, 21, 50), (165, 100, 85)),
    class_name[6] : ((60, 2, 25), (165, 20, 65))
}


def parse_args():
    parser = ag.ArgumentParser()
    parser.add_argument('-i',
                        '--input-image',
                        required=True,
                        dest='input',
                        type=str,
                        help='Image of a face to be processed')
    args = vars(parser.parse_args())
    return args


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_roi(image, shape, i, j):
    x, y, w, h = cv2.boundingRect(shape[np.newaxis,i:j])
    center = np.array([int(x + w/2), int(y + h/2)])
    return image[y:y+h, x:x+w], (x, y, w, h), center


def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
    hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False


def find_class(hsv):
    color_id = 7
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id


def map_to_cartoon(color):
    mapping = {"Blue": 1,
               "Blue Gray": 1,
               "Brown": 0,
               "Brown Gray": 0,
               "Brown Black": 4,
               "Green": 2,
               "Green Gray": 2, 
               "Other": 4}
    return mapping[color]


def get_eye_color(facePhoto):
    """
    facePhoto - image as a numpy array in BGR color shceme
    """
    
    # Converting to HSV:
    # facePhoto = cv2.imread(facePhotoPath)
    facePhoto = cv2.cvtColor(facePhoto, cv2.COLOR_BGR2HSV)
    height, width = facePhoto.shape[:2]
    
    PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat' # Rework later
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    imgMask = np.zeros((height, width, 1))
    
    # Facial landmarks detection
    fullImgBox = dlib.rectangle(0, 0, facePhoto.shape[0] - 1, facePhoto.shape[1] - 1)
    shape = predictor(facePhoto, fullImgBox)
    shape = shape_to_np(shape)
    
    # Getting ROIs
    leftEye, leCoords, leCenter = get_roi(facePhoto, shape, *FACIAL_LANDMARKS_IDXS['left_eye'])
    rightEye, reCoors, reCenter = get_roi(facePhoto, shape, *FACIAL_LANDMARKS_IDXS['right_eye'])
    
    # Computing some metrics
    eyeDistance = np.linalg.norm(leCenter - reCenter)
    eyeRadius = eyeDistance / 15 # approx
    # faceWidth = np.linalg.norm(shape[0] - shape[16])
    # eyeDistanceCoeff = eyeDistance / faceWidth
    
    # Forming mask
    cv2.circle(imgMask, tuple(leCenter), int(eyeRadius), (255,255,255), -1)
    cv2.circle(imgMask, tuple(reCenter), int(eyeRadius), (255,255,255), -1)
    
    eye_class = np.zeros(len(class_name), np.float)
    
    for y in range(0, height):
        for x in range(0, width):
            if imgMask[y, x] != 0:
                eye_class[find_class(facePhoto[y, x])] += 1
                
    main_color_index = np.argmax(eye_class[:len(eye_class)-1])
    total_vote = eye_class.sum()
    
    return map_to_cartoon(class_name[main_color_index])


if __name__ == '__main__':
    args = parse_args()
    color = get_eye_color(args['input'])
    print(color)