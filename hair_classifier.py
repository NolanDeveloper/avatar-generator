import cv2
import numpy as np
from evaluate import evaluate


def get_hair_color(image):
    hair_mask = evaluate(image)
    hair_mask = hair_mask.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    return cv2.cvtColor(np.array([[list(map(np.uint8, cv2.mean(image, mask=hair_mask[:h, :w])[:-1]))]]), cv2.COLOR_HSV2BGR)[0][0]


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
                dataset.append(get_hair_color(cv2.imread(root + "/" + filename)))
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