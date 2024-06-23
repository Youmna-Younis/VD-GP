import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    MODEL_WEIGHTS = os.path.join(basedir, 'model_weights.h5')
    MODEL_NEW = os.path.join(basedir, 'modelnew.h5')
    TOKEN_FILE = os.path.join(basedir, 'token.txt')
    YOLO_VIDEOS_MODEL = os.path.join(basedir, 'yolo_videos_model.pkl')
    YOLOV8N = os.path.join(basedir, 'yolov8n.pt')
