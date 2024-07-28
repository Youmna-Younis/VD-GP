from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")
