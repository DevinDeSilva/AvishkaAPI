import os
from flask import Flask, request,jsonify, make_response
from server import *
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = 'test'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif',"csv"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "Hello, Flask!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'data' not in request.files:
            return make_response("File not found", 400)
        file = request.files['data']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        
        print(file.filename,file, allowed_file(file.filename))
        if file.filename == '':
            return make_response("No selected file", 400)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_resuts = run_model(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return {
                "csv":file_resuts
            }
        return make_response("Error in files.", 400)
    else:
        return make_response("Use Post method.", 400)