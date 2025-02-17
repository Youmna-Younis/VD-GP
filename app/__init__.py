from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

def create_app():
    app = Flask(__name__, static_folder='static')
    load_dotenv()
    CORS(app)

    app.config['UPLOAD_FOLDER'] = './videos'

    with app.app_context():
        from . import routes

    return app
