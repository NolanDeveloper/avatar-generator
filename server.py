from flask import Flask, escape, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from PIL import Image
import io
import re
from matcher import find_best_match, load_database
import os

from get_eye_color import get_eye_color
from face_color_classifier import get_skin_color_number

app = Flask(__name__)
CORS(app)

cnt=0

DB_PATH="cartoonset10k"

@app.route('/pic', methods=['POST'])
def generate_avatar():
    content = request.json
    file_name = convert_base64_pic_to_ndarray(re.sub(r"data:image\/\w+;base64,", "", content['pic']))
    image = cv2.imread(file_name)
    eye_color = get_eye_color(image)
    skin_color = get_skin_color_number(image)
    is_male = content['genderValue'] == 'male'
    if content['genderValue']  == 'none':
        is_male = None
    cartoon_database = load_database(DB_PATH)
    params = { 'eye_color': eye_color, 'face_color': skin_color }
    pic_for_user = find_best_match(cartoon_database, params, not is_male)[:-3] + 'png'
    with open(pic_for_user, 'rb') as f:
        encoded_file = base64.b64encode(f.read())
        return encoded_file

def convert_base64_pic_to_ndarray(base64_pic):
    imgdata = base64.b64decode(base64_pic)
    with open('tmp__{}.jpg'.format(cnt), 'wb') as f:
        f.write(imgdata)
    return 'tmp__{}.jpg'.format(cnt)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5005)
