"""Code for our api app"""
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

@app.route('/strains', methods=['Post'])
def strains():
    """ a route, expects json object with 1 key """

    # receive input
    lines = request.get_json(force=True)

    # get data from json
    text = lines['input']  # json keys to be determined

    # validate input (optional)
    assert isinstance(text, str)

    # deserialize the pretrained model
    with open('medembedv2.pkl', 'rb') as mod:
        model = pickle.load(mod)

    # predict
    output = model.predict([text])

    # dictionary output for json
    send_back = {'prediction': output}

    # give output to sender.
    return jsonify(send_back)