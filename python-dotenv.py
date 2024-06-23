import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    basedir = os.path.abspath(os.path.dirname(__file__))
    MODEL_WEIGHTS = os.path.join(basedir, 'model_weights.h5')
    MODEL_NEW = os.path.join(basedir, 'modelnew.h5')
    TOKEN_FILE = os.getenv('TOKEN_FILE', os.path.join(basedir, 'token.txt'))
    YOLO_VIDEOS_MODEL = os.path.join(basedir, 'yolo_videos_model.pkl')
    YOLOV8N = os.path.join(basedir, 'yolov8n.pt')
