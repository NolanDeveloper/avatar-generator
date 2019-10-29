import glob
import csv
import random


PARAMETERS = [
    "facial_hair",
    "glasses",
    "glasses_color",
    "face_color",
    "hair_color",
    "eye_color",
    "face_shape",
    "hair",
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


def find_best_match(database, desired_parameters, is_male=None):
    if is_male is None:
        is_male = bool(random.getrandbits(1))
    desired_parameters['eye_lashes'] = int(not is_male)
    desired_parameters['face_shape'] = random.choice(GOOD_FACE_SHAPES)
    desired_parameters['glasses'] = 11
    desired_parameters['facial_hair'] = 14
    images = database.items()
    for parameter in PARAMETERS:
        if parameter in desired_parameters:
            new_mages = list(filter(lambda image: image[1][parameter] == desired_parameters[parameter], images))
            if not new_mages:
                break
            images = new_mages
    return random.choice(images)[0]


if __name__ == '__main__':
    print(find_best_match(load_database('./cartoonset10k'), {}))

