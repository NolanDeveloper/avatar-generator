import glob
import csv
import random

import cv2

PARAMETERS = [
    "facial_hair",
    "glasses",
    "hair",
    "eye_lashes",
    "face_color",
    "eye_color",
    "hair_color",
    "face_shape",
    "eye_angle",
    "eyebrow_shape",
    "eye_eyebrow_distance",
    "eye_slant",
    "eyebrow_width",
    "eye_lid",
    "chin_length",
    "eyebrow_weight",
    "eyebrow_thickness",
    "glasses_color"]


def load_database(db_path):
    database = {}
    pattern = '{}/*.csv'.format(db_path)
    for name in glob.glob(pattern):
        with open(name, 'r') as file:
            image = {}
            for row in csv.reader(file):
                image[row[0]] = int(row[1].strip())
            database[name] = image
    return database


GOOD_FACE_SHAPES = [1, 2, 3, 5, 6]
MALE_HAIRS = [10, 11, 17, 23, 31, 32, 37, 39, 43, 44, 45, 46, 49, 51, 52]
FEMALE_HAIRS = [3, 4, 6, 9, 10, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 40, 42, 47, 48]


def find_best_match(database, desired_parameters, is_male=None):
    if is_male is None:
        is_male = bool(random.getrandbits(1))
    desired_parameters['eye_lashes'] = int(is_male)
    desired_parameters['face_shape'] = random.choice(GOOD_FACE_SHAPES)
    desired_parameters['glasses'] = 11
    desired_parameters['facial_hair'] = 14
    desired_parameters['hair'] = random.choice(MALE_HAIRS if is_male else FEMALE_HAIRS)
    images = database.items()
    for parameter in PARAMETERS:
        if parameter in desired_parameters:
            new_images = list(filter(lambda image: image[1][parameter] == desired_parameters[parameter], images))
            if not new_images:
                break
            images = new_images
    return random.choice(images)[0]


if __name__ == '__main__':
    from get_eye_color import get_eye_color
    from face_color_classifier import get_skin_color_number
    from hair_classifier import get_hair_color_number
    image = cv2.imread('./CFD Version 2.0.3/CFD 2.0.3 Images/WM-238/CFD-WM-238-020-N.jpg')
    eye_color = get_eye_color(image)
    skin_color = get_skin_color_number(image)
    hair_color = get_hair_color_number(image)
    cartoon_database = load_database('./cartoonset10k')
    params = { 'eye_color': eye_color, 'face_color': skin_color, 'hair_color': hair_color }
    pic_for_user = find_best_match(cartoon_database, params, True)[:-3] + 'png'
    print(pic_for_user)