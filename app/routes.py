from flask import Blueprint, current_app

main = Blueprint('main', __name__)

@main.route('/')
def index():
    token_file = current_app.config['TOKEN_FILE']
    with open(token_file, 'r') as file:
        data = file.read()
    return data
