from flask import Flask, Response, render_template, request, jsonify, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_cors import CORS
from base64 import b64encode
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from keras.models import load_model
import cv2
import imageio
import logging
import numpy as np
import os
import pytz
import requests
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageFont, ImageDraw
from skimage.io import imread
from skimage.transform import resize
from ultralytics import YOLO

# App configuration
app = Flask(__name__, static_folder='static')
load_dotenv()
CORS(app)

# Logging configuration
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Constants
GAMMA = 0.67
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255 for i in np.arange(0, 256)]).astype("uint8")
UTC = pytz.timezone('Africa/Cairo')

# Load models
model_old = load_model('./CNN-LSTM.h5')
model_new = load_model('./CNN-LSTM.h5')
base_model = keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3), include_top=False, weights='imagenet', classes=2)
selected_model = model_old  # Default selected model

# Camera configuration
camera = cv2.VideoCapture(0)

# Form class
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

# Functions
def Alert(videopath):
    EG = datetime.now(UTC)
    current_date = EG.strftime("%d-%m-%y")
    current_time = EG.strftime("%H-%M-%S")
    filename = "token.txt"
    with open(filename, 'r') as file:
        telegram_auth_token = file.readline().strip()
        telegram_group_id = file.readline().strip()
    msg = f"Emergency: Violence detected\nDATE: {current_date}\nTIME: {current_time}\nPlease respond immediately"
    videofile = {'video': open(videopath, 'rb')}
    telegramAPIURL = f"https://api.telegram.org/bot{telegram_auth_token}/sendMessage?chat_id=@{telegram_group_id}&text={msg}"
    telegramRespose = requests.get(telegramAPIURL)
    resp = requests.post(f"https://api.telegram.org/bot{telegram_auth_token}/sendVideo?chat_id=@{telegram_group_id}", files=videofile)
    if telegramRespose.status_code == 200:
        print("Message has been sent successfully!")
    else:
        print("ERROR!")

def convert_avi_to_mp4(avi_filepath, mp4_filepath):
    reader = imageio.get_reader(avi_filepath)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(mp4_filepath, fps=fps)
    for frame in reader:
        writer.append_data(frame)
    writer.close()

def video_reader(cv2, filename):
    frames = np.zeros((30, 160, 160, 3), dtype=float)
    vid = cv2.VideoCapture(filename)
    if vid.isOpened():
        grabbed, frame = vid.read()
    else:
        grabbed = False
    frm = resize(frame, (160, 160, 3))
    frm = np.expand_dims(frm, axis=0)
    if np.max(frm) > 1:
        frm = frm / 255.0
    frames[0][:] = frm
    i = 1
    while i < 30:
        grabbed, frame = vid.read()
        frm = resize(frame, (160, 160, 3))
        frm = np.expand_dims(frm, axis=0)
        if np.max(frm) > 1:
            frm = frm / 255.0
        frames[i][:] = frm
        i += 1
    return frames

def create_pred_imgarr(base_model, video_frm_ar):
    video_frm_ar_dim = np.zeros((1, 30, 160, 160, 3), dtype=float)
    video_frm_ar_dim[0][:][:] = video_frm_ar
    pred_imgarr = base_model.predict(video_frm_ar)
    pred_imgarr = pred_imgarr.reshape(1, pred_imgarr.shape[0], 5 * 5 * 1024)
    return pred_imgarr

def pred_fight(model, pred_imgarr):
    pred_test = model.predict(pred_imgarr)
    return pred_test[0][1], pred_test[0][0]

def yolov8n(video_path, output_video_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open input video file")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if output_video_path.endswith('.mp4'):
        output_video_path = output_video_path[:-4] + '.avi'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gamma_corrected_frame = cv2.LUT(frame, gamma_table)
        results = model(gamma_corrected_frame)
        black_frame = np.zeros_like(frame)
        for i in range(len(results[0].boxes.cls)):
            if results[0].boxes.cls[i] == 0:
                x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i][:4])
                person_region = frame[y1:y2, x1:x2]
                black_frame[y1:y2, x1:x2] = person_region
        out.write(black_frame)
    cap.release()
    out.release()
    return output_video_path

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    global selected_model
    video_file = request.files['video_file']
    video_path = "./videos/" + video_file.filename
    if 'modelSelection' in request.form:
        selected_model = request.form['modelSelection']
    file_extension = os.path.splitext(video_file.filename)[-1].lower()
    if file_extension not in ['.mp4', '.avi']:
        raise ValueError("Unsupported video file format. Supported formats: mp4, avi")
    if file_extension == '.avi':
        mp4_filepath = "./videos/" + 'converted_video.mp4'
        convert_avi_to_mp4(video_path, mp4_filepath)
        video_path = mp4_filepath
    video_frm_ar = video_reader(cv2, video_path)
    pred_imgarr = create_pred_imgarr(base_model, video_frm_ar)
    probability_violence, probability_non_violence = 0.0, 0.0
    if selected_model == 'oldModel':
        model_to_use = model_old
        probability_violence, probability_non_violence = pred_fight(model_to_use, pred_imgarr)
    else:
        p = Yolo2predictions(video_path, r'/temp.avi')
        probability_violence = p[0]
        probability_non_violence = 1 - p[0]
    mime_type = 'video/mp4' if file_extension == '.mp4' else 'video/x-msvideo'
    video = open(video_path, 'rb').read()
    src = f'data:{mime_type};base64,' + b64encode(video).decode()
    video_tag = f'<video width="800" height="600" controls><source src="{src}" type="{mime_type}"></video>'
    isviolence = probability_violence - probability_non_violence
    if isviolence >= 0:
        Alert(video_path)
    return render_template('index.html', prob_violence=probability_violence, prob_non_violence=probability_non_violence, video_tag=video_tag, selected_model=selected_model)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    input_path = 0
    output_path = "output.mp4"
    fps = 30
    vid = cv2.VideoCapture(input_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f'fps: {fps}')
    writer = None
    (W, H) = (None, None)
    i = 0
    Q = deque(maxlen=128)
    video_frm_ar = np.zeros((1, int(fps), 160, 160, 3), dtype=float)
    frame_counter = 0
    frame_list = []
    preds = None
    maxprob = None
    AlertIsSend = 0
    while True:
        frame_counter += 1
        grabbed, frm = vid.read()
        if not grabbed:
            print('There is no frame. Streaming ends.')
            break
        if fps != 30:
            print('Please set fps=30')
            break
        if W is None or H is None:
            (H, W) = frm.shape[:2]
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        rgb = resize(rgb, (160, 160))
        if np.max(rgb) > 1:
            rgb = rgb / 255.0
        Q.append(rgb)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)
        if frame_counter % 30 == 0:
            video_frm_ar[0][:][:] = np.array(Q)
            pred_imgarr = create_pred_imgarr(base_model, video_frm_ar[0])
            probability_violence, probability_non_violence = pred_fight(model_old, pred_imgarr)
            preds = probability_violence - probability_non_violence
            maxprob = max(probability_violence, probability_non_violence)
            frame_counter = 0
        if preds is not None and maxprob is not None:
            text = 'Violence Probability: %.2f' % (probability_violence)
            Y = H - 50
            cv2.putText(frm, text, (10, Y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 5)
            frame_list.append(frm)
        if preds >= 0.3:
            if AlertIsSend == 0:
                print('Alert')
                output_video = output_path[:-4] + 'AVI'
                yolov8n(output_path, output_video)
                Alert(output_video)
                AlertIsSend = 1
        writer.write(frm)
    print('[INFO] Cleaning up...')
    writer.release()
    vid.release()
    return "Streaming ended."

@app.route('/upload', methods=['POST'])
def upload_file():
    form = UploadFileForm()
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        form.file.data.save(file_path)
        flash(f'File successfully uploaded: {filename}', 'success')
        return redirect(url_for('index'))
    flash('File upload failed', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
