from flask import current_app as app, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from .forms import UploadFileForm
from .models import load_models, Alert, convert_avi_to_mp4, video_reader, create_pred_imgarr, pred_fight, yolov8n
import os
from base64 import b64encode
import cv2
import numpy as np

model_old, model_new, base_model = load_models()
selected_model = model_old

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    global selected_model
    video_file = request.files['video_file']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)

    if 'modelSelection' in request.form:
        selected_model = request.form['modelSelection']

    file_extension = os.path.splitext(video_file.filename)[-1].lower()
    if file_extension not in ['.mp4', '.avi']:
        raise ValueError("Unsupported video file format. Supported formats: mp4, avi")
    
    if file_extension == '.avi':
        mp4_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'converted_video.mp4')
        convert_avi_to_mp4(video_path, mp4_filepath)
        video_path = mp4_filepath

    video_frm_ar = video_reader(cv2, video_path)
    pred_imgarr = create_pred_imgarr(base_model, video_frm_ar)

    if selected_model == 'oldModel':
        model_to_use = model_old
        probability_violence, probability_non_violence = pred_fight(model_to_use, pred_imgarr)
    else:
        probability_violence, probability_non_violence = pred_fight(model_new, pred_imgarr)

    mime_type = 'video/mp4' if file_extension == '.mp4' else 'video/x-msvideo'
    video = open(video_path, 'rb').read()
    src = f'data:{mime_type};base64,' + b64encode(video).decode()
    video_tag = f'<video width="800" height="600" controls><source src="{src}" type="{mime_type}"></video>'
    
    if probability_violence > probability_non_violence:
        Alert(video_path)

    return render_template('index.html', prob_violence=probability_violence, prob_non_violence=probability_non_violence, video_tag=video_tag, selected_model=selected_model)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    # (Implementation for the streaming route)
    pass

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
