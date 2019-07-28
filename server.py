from flask import Flask, render_template, request, redirect, url_for, send_from_directory, json, jsonify
from datetime import datetime
from translate import Translator
import numpy as np
import cv2
import os
import random
import time
import string
import shutil
import sys
sys.path.append('src')
import nn_process


SAVE_DIR = "./images"
translator = Translator(from_lang = "en", to_lang = "ja")

app = Flask(__name__, static_url_path="")


def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])


@app.route('/')
def index():
    global extract_feature
    global generate_poem
    extract_feature = nn_process.create('extract_feature')
    generate_poem = nn_process.create('generate_poem')
    return render_template('index.html')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

@app.route('/upload', methods=['POST'])
def upload():
    shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)

    if request.files['image']:
        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        # 保存
        dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_") + random_str(5)
        save_path = os.path.join(SAVE_DIR, dt_now + ".jpg")
        cv2.imwrite(save_path, img)

        # img2poem
        img_feature = extract_feature(save_path)
        poems = generate_poem(img_feature)[0].split('\n')
        for i in range(len(poems)):
            poems[i] = translator.translate(poems[i])

        return render_template('index.html', images=os.listdir(SAVE_DIR)[::-1], poems=poems)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)
