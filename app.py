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


def getEmotion(message):
    class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

    with open('./model/tokenizer.pickle', 'rb') as handle:
        tokenizerX = pickle.load(handle)

    max_seq_len = 500
    seq = tokenizerX.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    predX = app.predictor.predict(padded)

    return {"predicted": class_names[np.argmax(predX)]}


@app.route('/predict', methods=['POST'])
def predict():
    data = {'Success': False}
    # get the request params
    params = request.json
    if (params == None):
        params = request.args
    if (params != None):
        message = params.get('message')
        data['response'] = getEmotion([message])
        data['Success'] = True
    return jsonify(data)


def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()
