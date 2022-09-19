from flask import Flask
from flask import request
from flask_cors import CORS
from PIL import Image
from classifier import Classifier

cls = Classifier('classifier.trch')

app = Flask(__name__)

CORS(app)

@app.post("/")
def handle_post():
    """
    Handles the POST request from the frontend and returns the prediction.
    """
    if 'file' not in request.files:
        return {'status': True, 'message': 'Error',  'payload': response_model}

    file = request.files['file']
    file = Image.open(file)


    response_model = cls.classify(file)
    return {'status': True, 'message': 'OK',  'payload': response_model}

@app.get("/")
def handle_get():
    """
    Handles the GET request
    """
    return 'GET'
    