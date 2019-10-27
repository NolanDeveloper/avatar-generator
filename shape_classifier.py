import pickle
from pathlib import Path

import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans


def get_shape_points(picture, detector, predictor):
    faces = detector(picture)
    if len(faces) < 1:
        return None
    else:
        landmarks = predictor(picture, faces[0])
        return [[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 16)]


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_mean_angle(picture, detector, predictor):
    shape_points = get_shape_points(picture, detector, predictor)
    angles = [angle_between(shape_points[i], shape_points[i + 1]) for i in range(len(shape_points) - 1)]
    mean_angle = np.mean(angles)
    return mean_angle


def flatten(v):
    return [item for sublist in v for item in sublist]


def get_feature_vector(shape_points):
    return flatten(shape_points)


def create_classifier(pictures, detector, predictor):
    X = [get_feature_vector(get_shape_points(picture, detector, predictor)) for picture in pictures]
    kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
    return kmeans


def load_pictures(root_dir):
    file_names = []
    for cur_dir in Path(root_dir).glob('*'):
        if not cur_dir.is_dir():
            continue
        if not str(cur_dir).find("WM") or not str(cur_dir).find("WF"):
            continue
        files = list(cur_dir.glob('*.jpg'))
        if len(files) > 0:
            file_names.append(str(files[0]))
    return [cv2.imread(filename, 0) for filename in file_names[:50]]


def classify_shape(filename):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    classifier = pickle.load(open('classifier', 'r'))
    picture = cv2.imread(filename, 0)
    features = get_feature_vector(get_shape_points(picture, detector, predictor))
    return classifier.predict(features)
