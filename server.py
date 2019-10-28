from flask import Flask, escape, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from PIL import Image
import io
import re

app = Flask(__name__)
CORS(app)

@app.route('/pic', methods=['POST'])
def hello():
    content = request.json
    ndarr = convert_base64_pic_to_ndarray(re.sub(r"data:image\/\w+;base64,", "", content['pic']))
    return content['pic']

def convert_base64_pic_to_ndarray(base64_pic):
    imgdata = base64.b64decode(base64_pic)
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5005)