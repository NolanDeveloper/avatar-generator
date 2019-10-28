import glob
import csv


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
    for name in glob.glob('{}/*.csv'.format(db_path)):
        with open(name, 'r') as file:
            image = {}
            for row in csv.reader(file):
                image[row[0]] = int(row[1].strip())
            database[name] = image
    return database


def find_best_match(database, desired_parameters):
    best_name = None
    best_distance = None
    for name, image in database.items():
        distance = list(map(lambda parameter: image[parameter] != desired_parameters.get(parameter, 0), PARAMETERS))
        if not best_distance or distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name


if __name__ == '__main__':
    print(find_best_match(load_database(), {'facial_hair': 2}))

