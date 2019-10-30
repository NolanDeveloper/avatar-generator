import cv2
import numpy as np
from evaluate import evaluate


import math


HAIR_COLORS = {
    0: (43, 45.9, 100),
    1: (33, 69.8, 100),
    2: (20, 80.2, 92.9),
    3: (13, 78.1, 75.3),
    4: (38, 76.9, 84.7),
    5: (18, 68.9, 63.1),
    6: (22, 69.2, 35.7),
    7: (6, 33.3, 11.8),
    8: (200, 17.2, 68.2),
    9: (0, 0, 87.8),
}


def to_real_hsv(hsv):
    return hsv[0] / 255 * 360, hsv[1] / 255 * 100, hsv[2] / 255 * 100


def nearest_color(color):
    minDist = None
    bestNumber = None
    for number, hair_color in HAIR_COLORS.items():
        dist = 0
        for i in range(3):
            dist += (float(color[i]) - hair_color[i]) ** 2
        dist = math.sqrt(dist)
        if minDist is None or dist < minDist:
            minDist = dist
            bestNumber = number
    return bestNumber


def get_hair_color(image):
    hair_mask = evaluate(image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    return cv2.mean(image, mask=hair_mask[:h, :w])[:-1]


def get_hair_color_number(image):
    color = to_real_hsv(get_hair_color(image))
    return nearest_color(color)


if __name__ == '__main__':
    # # image = cv2.imread('./data/Figaro1k/Original/Testing/Frame00010-org.jpg')
    # # image = cv2.imread('./data/Figaro1k/Original/Testing/Frame00032-org.jpg')
    # image = cv2.imread('./data/Figaro1k/Original/Testing/Frame00458-org.jpg')
    # # image = cv2.imread('./data/Figaro1k/Original/Testing/A.jpg')
    # color = get_hair_color(image)
    # color_image = np.array([[color] * 100] * 100)
    # cv2.imshow("Image", color_image)
    # cv2.waitKey(0)

    import sys
    import sklearn.cluster as sk
    import os

    dataset = []
    for root, _, files in os.walk('data'):
        for filename in files:
            if filename.endswith('.jpg'):
                dataset.append(get_hair_color_number(cv2.imread(root + "/" + filename)))
                if not (len(dataset) % 10):
                    print(len(dataset))
                if len(dataset) == 100:
                    break
        if len(dataset) == 100:
            break
    model = sk.KMeans(n_clusters=15).fit(dataset)
    for i in range(15):
        # os.mkdir(str(i))
        center = model.cluster_centers_[i]
        image = np.array([[center] * 100] * 100)
        cv2.imwrite(str(i) + '.jpg', image)

    # pickle.dump(model, open('model.dat', 'wb'))