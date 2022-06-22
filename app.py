from email import message
import json
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
import numpy as np
from keras import backend
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import time

app = Flask(__name__)


@app.before_first_request
def load_model_to_app():
    app.predictor = load_model('./model/cnn_w2v.h5')


@app.route("/")
def index():
    return "HELLO WORLD"


def get_emotion(message):
    seq = tokenizerX.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    predX = app.predictor.predict(padded)
    return class_names[np.argmax(predX)]


def get_percentage(message):
    seq = tokenizerX.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    predX = app.predictor.predict(padded)
    dictionary = dict(zip(class_names, predX[0]*100))
    for key, value in dictionary.items():
        dictionary[key] = str(value)
    return dictionary


@app.route('/predict', methods=['POST'])
def predict():
    data = {'Success': False}
    params = request.json
    if (params == None):
        params = request.args
    if (params != None):
        message = params.get('message')
        data['emotion'] = get_emotion([message])
        data['Success'] = True
    return jsonify(data)


@app.route('/getPercentage', methods=['POST'])
def percentage():
    data = {'Success': False}
    params = request.json
    if (params == None):
        params = request.args
    if (params != None):
        message = params.get('message')
        data['emotion'] = get_percentage([message])
        data['Success'] = True
    return jsonify(data)


with open('./model/tokenizer.pickle', 'rb') as handle:
    tokenizerX = pickle.load(handle)
max_seq_len = 500
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']


def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()
