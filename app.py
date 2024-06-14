from flask import Flask, Response, render_template, request, jsonify, flash
import cv2  # openCV 4.5.1
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import time
import subprocess
import imageio
from base64 import b64encode

from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageFont, ImageDraw  # add caption by using custom font

from collections import deque
from tensorflow import keras

###farah model
from keras.models import load_model
import matplotlib.pyplot as plt

from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import tqdm
import sys
import logging

app = Flask(__name__, static_folder='static')



from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import tqdm
import sys
import logging
from Model_VD import Yolo2predictions,video_reader,create_pred_imgarr
from Model_VD import pred_fight,model_old,convert_avi_to_mp4,base_model
app = Flask(__name__, static_folder='static')


########################################################################3
@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    global selected_model

    video_file = request.files['video_file']
    video_path = "./videos/" + video_file.filename

    if 'modelSelection' in request.form:
        selected_model = request.form['modelSelection']

    # Check if the file extension is supported
    file_extension = os.path.splitext(video_file.filename)[-1].lower()
    if file_extension not in ['.mp4', '.avi']:
        raise ValueError("Unsupported video file format. Supported formats: mp4, avi")

    if file_extension == '.avi':
        # Convert AVI to MP4
        mp4_filepath = "./videos/" + 'converted_video.mp4'
        convert_avi_to_mp4(video_path, mp4_filepath)
        video_path = mp4_filepath

    video_frm_ar = video_reader(cv2, video_path)
    pred_imgarr = create_pred_imgarr(base_model, video_frm_ar)

    ####
    probability_violence = 0.0
    probability_non_violence = 0.0
    if selected_model == 'oldModel':
        model_to_use = model_old
        probability_non_violence, probability_violence = pred_fight(model_to_use, pred_imgarr)


    else:
        p = Yolo2predictions(video_path, r'D:/books/GP/violence/photo/peacekeeper/videos/temp.avi')
        probability_violence = p[0][0]
        probability_non_violence = 1 - p[0][0]

    # Determine the MIME type based on file extension
    if file_extension == '.mp4':
        mime_type = 'video/mp4'
    else:
        mime_type = 'video/x-msvideo'

    # Generate the video tag
    video = open(video_path, 'rb').read()
    mime_type = 'video/mp4'  # Use MP4 MIME type
    src = f'data:{mime_type};base64,' + b64encode(video).decode()
    video_tag = f'<video width="800" height="600" controls><source src="{src}" type="{mime_type}"></video>'

    return render_template('index.html', prob_violence=probability_violence, prob_non_violence=probability_non_violence,
                           video_tag=video_tag, selected_model=selected_model)


@app.route('/start_stream', methods=['POST'])
def start_stream():
    input_path = 0
    output_path = "output.mp4"
    fps = 30
    vid = cv2.VideoCapture(input_path)
    fps = vid.get(cv2.CAP_PROP_FPS)  # recognize frames per secone(fps) of input_path video file.
    print(f'fps : {fps}')  # print fps.

    writer = None
    (W, H) = (None, None)
    i = 0  # number of seconds in video = The number of times that how many operated while loop .
    Q = deque(maxlen=128)

    video_frm_ar = np.zeros((1, int(fps), 160, 160, 3), dtype=float)  # frames
    frame_counter = 0  # frame number in 1 second. 1~30
    frame_list = []
    preds = None
    maxprob = None

    # . While loop : Until the end of input video, it read frame, extract features, predict violence True or False.
    # ----- Reshape & Save frame img as (30, 160, 160, 3) Numpy array  -----
    while True:
        frame_counter += 1
        grabbed, frm = vid.read()  # read each frame img. grabbed=True, frm=frm img. ex: (240, 320, 3)

        if not grabbed:
            print('There is no frame. Streaming ends.')
            break

        if fps != 30:
            print('Please set fps=30')
            break

        if W is None or H is None:  # W: width, H: height of frame img
            (H, W) = frm.shape[:2]

        output = frm.copy()  # It is necessary for streaming captioned output video, and to save that.

        frame = resize(frm, (160, 160, 3))  # > Resize frame img array to (160, 160, 3)
        frame_list.append(frame)  # Append each frame img Numpy array : element is (160, 160, 3) Numpy array.

        if frame_counter >= fps:  # fps=30 et al
            # . ----- we'll predict violence True or False every 30 frame -----
            # . ----- Insert (1, 30, 160, 160, 3) Numpy array to LSTM model ---
            # . ----- We'll renew predict result caption on output video every 1 second. -----
            # 30-element-appended list -> Transform to Numpy array -> Predict -> Initialize list (repeat)
            frame_ar = np.array(frame_list, dtype=np.float16)  # > (30, 160, 160, 3)
            frame_list = []  # Initialize frame list when frame_counter is same or exceed 30, after transforming to Numpy array.

            if (np.max(frame_ar) > 1):  # Scaling RGB value in Numpy array
                frame_ar = frame_ar / 255.0

            pred_imgarr = base_model.predict(
                frame_ar)  # > Extract features from each frame img by using MobileNet. (30, 5, 5, 1024)
            pred_imgarr_dim = pred_imgarr.reshape(1, pred_imgarr.shape[0], 5 * 5 * 1024)  # > (1, 30, 25600)

            preds = model_old.predict(
                pred_imgarr_dim)  # > (True, 0.99) : (Violence True or False, Probability of Violence)
            print(f'preds:{preds}')
            Q.append(preds)  # > Deque Q

            # Predict Result : Average of Violence probability in last 5 second
            if i < 5:
                results = np.array(Q)[:i].mean(axis=0)
            else:
                results = np.array(Q)[(i - 5):i].mean(axis=0)

            print(f'Results = {results}')  # > ex : (0.6, 0.650)

            maxprob = np.max(results)  # > Select Maximum Probability
            print(f'Maximum Probability : {maxprob}')
            print('')

            rest = 1 - maxprob  # Probability of Non-Violence
            diff = maxprob - rest  # Difference between Probability of Violence and Non-Violence's
            th = 100

            if diff > 0.80:
                th = diff  # ?? What is supporting basis?

            frame_counter = 0  # > Initialize frame_counter to 0
            i += 1  # > 1 second elapsed

            # When frame_counter>=30, Initialize frame_counter to 0, and repeat above while loop.

        # ----- Setting caption option of output video -----
        # Renewed caption is added every 30 frames(if fps=30, it means 1 second.)
        font1 = ImageFont.load_default()
        font2 = ImageFont.load_default()

        if preds is not None and maxprob is not None:
            if preds[0][0] >= preds[0][1]:  # > if violence probability < th, Violence=False (Normal, Green Caption)
                text1_1 = 'Normal'
                text1_2 = '{:.2f}%'.format(100 - (maxprob * 100))
                img_pil = Image.fromarray(output)
                draw = ImageDraw.Draw(img_pil)
                draw.text((int(0.025 * W), int(0.025 * H)), text1_1, font=font1, fill=(0, 255, 0, 0))
                draw.text((int(0.025 * W), int(0.095 * H)), text1_2, font=font2, fill=(0, 255, 0, 0))
                output = np.array(img_pil)

            else:  # > if violence probability > th, Violence=True (Violence Alert!, Red Caption)
                text2_1 = 'Violence Alert!'
                text2_2 = '{:.2f}%'.format(maxprob * 100)
                img_pil = Image.fromarray(output)
                draw = ImageDraw.Draw(img_pil)
                draw.text((int(0.025 * W), int(0.025 * H)), text2_1, font=font1, fill=(0, 0, 255, 0))
                draw.text((int(0.025 * W), int(0.095 * H)), text2_2, font=font2, fill=(0, 0, 255, 0))
                output = np.array(img_pil)

                # Save captioned video file by using 'writer'
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

        cv2.imshow('This is output', output)  # View output in new Window.
        writer.write(output)  # Save output in output_path

        key = cv2.waitKey(round(1000 / fps))  # time gap of frame and next frame
        if key == 27:  # If you press ESC key, While loop will be breaked and output file will be saved.
            print('ESC is pressed. Video recording ends.')
            break

    print('Video recording ends. Release Memory.')  # Output file will be saved.
    writer.release()
    vid.release()
    cv2.destroyAllWindows()
    return render_template("index.html")


if __name__ == '__main__':
    app.run(port=1000, debug=True)
