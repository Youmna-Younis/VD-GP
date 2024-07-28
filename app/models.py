from keras.models import load_model
from tensorflow import keras
import cv2
import numpy as np
import pytz
from datetime import datetime
import requests
import os
from skimage.transform import resize
import imageio

UTC = pytz.timezone('Africa/Cairo')

def load_models():
    model_old = load_model('./CNN-LSTM.h5')
    model_new = load_model('./CNN-LSTM.h5')
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3), include_top=False, weights='imagenet', classes=2)
    return model_old, model_new, base_model

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
    from ultralytics import YOLO
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
        black_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf.cpu().numpy()[0]
                if conf >= 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    cv2.rectangle(black_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(black_frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        output_frame = cv2.addWeighted(frame, 0.5, black_frame, 0.5, 0)
        out.write(output_frame)
    cap.release()
    out.release()
    print(f"Output video saved as: {output_video_path}")
