import cv2
import numpy as np
import pickle


def hsv(color):
    h = color[0] / 255 * 360
    s = color[1] / 255 * 100
    v = color[2] / 255 * 100
    return h, s, v

def hsv_to_bgr(hsv):
    r = hsv[0] / 360 * 255
    g = hsv[1] / 100 * 255
    b = hsv[2] / 100 * 255
    return r, g, b


def get_skin_color(image):
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img, lower_threshold, upper_threshold)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    return cv2.cvtColor(np.array([[list(map(np.uint8, cv2.mean(img, mask=skin_mask)[:-1]))]]), cv2.COLOR_HSV2BGR)[0][0]


CORRESPONDENCE = {
    2: 0,
    4: 1,
    5: 2,
    0: 3,
    10: 4,
    7: 5,
    6: 6,
    11: 7,
    3: 8,
    9: 9,
    1: 10,
    8: 11,
}


def get_skin_color_number(image):
    model = pickle.load(open('model.dat', 'rb'))
    color = get_skin_color(image)
    n = model.predict(np.array(color).reshape(1, -1))[0]
    return CORRESPONDENCE[n]


# if __name__ == '__main__':
#     import sys
#     import sklearn.cluster as sk
#
#     dataset = []
#     for root, _, files in os.walk('CFD Version 2.0.3'):
#         for filename in files:
#             if filename.endswith('.jpg'):
#                 dataset.append(get_skin_color(cv2.imread(root + "/" + filename)))
#         print(len(dataset))
#     model = sk.KMeans(n_clusters=12).fit(dataset)
#     for i in range(12):
#         # os.mkdir(str(i))
#         center = model.cluster_centers_[i]
#         image = np.array([[center] * 100] * 100)
#         cv2.imwrite(str(i) + '.jpg', image)
#
#     pickle.dump(model, open('model.dat', 'wb'))

    # for root, _, files in os.walk('CFD Version 2.0.3'):
    #     for filename in files:
    #         if filename.endswith('.jpg'):
    #             path = root + "/" + filename
    #             color = get_skin_color(cv2.imread(path))
    #             n = model.predict(np.array(color).reshape(1, -1))[0]
    #             shutil.copy(path, str(n) + "/" + filename)

    # print(list(map(lambda x: x.tolist(), sorted(model.cluster_centers_, key=lambda x: x[2]))))
    # asianColor = model.predict(np.array(get_skin_color(cv2.imread('CFD-AF-200-228-N.jpg'))).reshape(1, -1))
    # africanColor = model.predict(np.array(get_skin_color(cv2.imread('CFD-BF-239-180-N.jpg'))).reshape(1, -1))
    # print(classify_skin_color(cv2.imread(sys.argv[1])))
