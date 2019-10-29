import glob
import csv
import random


PARAMETERS = [
    "hair",
    "facial_hair",
    "glasses",
    "glasses_color",
    "face_color",
    "hair_color",
    "eye_color",
    "face_shape",
    "eye_angle",
    "eyebrow_shape",
    "eye_eyebrow_distance",
    "eye_lashes",
    "eye_slant",
    "eyebrow_width",
    "eye_lid",
    "chin_length",
    "eyebrow_weight",
    "eyebrow_thickness"]


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
    print(find_best_match(load_database('./cartoonset10k'), {}))

