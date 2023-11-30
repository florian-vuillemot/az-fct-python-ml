"""
Simple Prediction Service from a local model.

This script essentially sets up a simple web service that accepts input data via POST requests, uses a local pre-trained model to make predictions based on this data, and returns the predictions.
"""
import pickle

import azure.functions as func
from flask import Flask, request

app = Flask(__name__)


# Retrieving the model.
with open('model.pickle', 'rb') as pickle_file:
    # Loading the bytes field model from the file.
    model_bytes = pickle_file.read()
    # Converting the bytes filed model to a usable Python object.
    clf = pickle.loads(model_bytes)


@app.route("/", methods=["POST"])
def predict():
    # Retrieving the data to predict.
    # For education purposes, there is no input validation.
    inputs = request.get_json()

    # Predict the incomming value.
    res = clf.predict(inputs)
    return res.tolist()


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return func.WsgiMiddleware(app).handle(req, context)
