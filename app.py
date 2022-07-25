from flask import Flask, render_template, request
from werkzeug.exceptions import BadRequest
import os
from werkzeug.utils import secure_filename #3.8
import json
from flask import send_from_directory
#from flask_restplus import abort
#from process import classification
from hand import recognization
import recognise
import logging
from logging.handlers import RotatingFileHandler
import time
import traceback
import random

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.getcwd() + '/uploads'
app.config['FACE_FOLDER'] = os.getcwd() + '/uploads'


@app.route('/')
def index():
  return "It is working"

'''

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('[Info] In /upload with method=Post')
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        result = classification(path)
        os.remove(path)
        return json.dumps({'result':result})
    else:
        return "This is for Post request only. Try a POst request"
'''


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('[Info] In /upload with method=Post')
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['FACE_FOLDER'], filename)
        file.save(path)
        result = recognise.recognizeImage(path)
        print('aaaa')
        #os.remove(path)
        result=int(result)
        if result==0 or result==1:
          result="Real"
        else:
          result="Fake"
        print(type(result))
        return json.dumps({'result':result})
    else:
        return "This is for Post request only. Try a Post request gg"

      
@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        print('[Info] In /upload with method=Post')
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['FACE_FOLDER'], filename)
        file.save(path)
        result = recognise.recognizeImage(path)
        #os.remove(path)
        return json.dumps({'result':result})
    else:
        return "This is for Post request only. Try a POst request"

@app.route('/number', methods=['GET', 'POST'])
def number():
    if request.method == 'POST':
        print('[Info] In /upload with method=Post')
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['HAND_FOLDER'], filename)
        file.save(path)
        result = recognization(path)
        os.remove(path)
        return json.dumps({'result':result})
    else:
        return "This is for Post request only. Try a POst request"
        


@app.after_request
def after_request(response):
    if response.status_code != 500:
        ts = time.strftime('[%Y-%b-%d %H:%M]')
    return response

@app.errorhandler(Exception)
def exceptions(e):
    ts = time.strftime('[%Y-%b-%d %H:%M]')
    tb = traceback.format_exc()
    return "Internal Server Error", 500

@app.errorhandler(404)
def Internalerror(e):
    return "page not found. %s"%(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0')  
