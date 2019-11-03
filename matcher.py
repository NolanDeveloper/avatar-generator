import glob
import csv
import random
from typing import Tuple

import cv2
import os

PARAMETERS = [
    "facial_hair",
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
    "glasses",
    "eyebrow_weight",
    "eyebrow_thickness",
    "glasses_color"]
GOOD_FACE_SHAPES = [1, 2, 3, 5, 6]
MALE_HAIRS = [10, 11, 17, 23, 31, 32, 37, 39, 43, 44, 45, 46, 49, 51, 52]
FEMALE_HAIRS = [3, 4, 6, 9, 10, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 40, 42, 47, 48]

NGENDERS = 2
NFACE_COLORS = 11
NEYE_COLORS = 5
NHAIR_COLORS = 10


class DecisionTree:
    def __init__(self):
        self.data = {}

    def add(self, item: Tuple):
        dictionary = self.data
        for i in range(len(item) - 1):
            if item[i] not in dictionary:
                dictionary[item[i]] = {} if i != len(item) - 2 else set()
            dictionary = dictionary[item[i]]
        dictionary.add(item[-1])

    def get(self, item: Tuple):
        dictionary = self.data
        for i in range(len(item)):
            if item[i] not in dictionary:
                return set()
            dictionary = dictionary[item[i]]
        return dictionary


def load_database(db_path):
    male_database = []
    female_database = []
    for root, _, files in os.walk(db_path):
        for filename in files:
            if not filename.endswith('.csv'):
                continue
            path = root + "/" + filename
            with open(path, 'r') as file:
                image = {}
                for row in csv.reader(file):
                    image[row[0]] = int(row[1].strip())
                if image['face_shape'] not in GOOD_FACE_SHAPES:
                    continue
                if image['facial_hair'] != 14:
                    continue
                is_male = image['eye_lashes'] and image['hair'] in MALE_HAIRS
                is_female = not image['eye_lashes'] and image['hair'] in FEMALE_HAIRS
                if is_male:
                    male_database.append((path, image))
                if is_female:
                    female_database.append((path, image))
    return female_database, male_database


def find_best_match(database, face_color, eye, hair, is_male=None):
    if is_male is None:
        is_male = bool(random.getrandbits(1))
    database = database[int(is_male)]
    database = filter(lambda image: image[1]['eye_color'] == eye and image[1]['hair_color'] == hair, database)
    result = min(database, key=lambda image: abs(image[1]['face_color'] - face_color))[0]
    return result[:-3] + 'png'


if __name__ == '__main__':
    from get_eye_color import get_eye_color
    from face_color_classifier import get_skin_color_number
    from hair_classifier import get_hair_color_number
    import time
    import pickle
    # before = time.monotonic()
    # cartoon_database = load_database('./cartoonset100k')
    # pickle.dump(cartoon_database, open('cartoon_database.dat', 'wb'))
    # print(time.monotonic() - before)
    # print("Done!")
    cartoon_database = pickle.load(open('cartoon_database.dat', 'rb'))
    # best = find_best_match(cartoon_database, 0, 0, 0)
    # print(best)
    image = cv2.imread('./tests/black-woman.jpg')
    eye_color = get_eye_color(image)
    print(f'eye color: {eye_color}')
    skin_color = get_skin_color_number(image)
    print(f'skin color: {skin_color}')
    hair_color = get_hair_color_number(image)
    print(f'hair color: {hair_color}')
    pic_for_user = find_best_match(cartoon_database, skin_color, eye_color, hair_color, False)
    print(pic_for_user)
