"""
Simple Prediction Service from a local model.

This script essentially sets up a simple web service that accepts input data via POST requests, uses a local pre-trained model to make predictions based on this data, and returns the predictions.
"""
import logging
import pickle

import azure.functions as func


# Create the Azure Function App.
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Retrieving the model.
with open('model.pickle', 'rb') as pickle_file:
    # Loading the bytes field model from the file.
    model_bytes = pickle_file.read()
    # Converting the bytes filed model to a usable Python object.
    clf = pickle.loads(model_bytes)


@app.route(route="predict")
def predict(req: func.HttpRequest):
    # Retrieving the data to predict.
    # For education purposes, there is no input validation.
    logging.info("enter function")
    inputs = req.get_json()
    logging.info(str(inputs))
    # Predict the incomming value.
    try:
        res = clf.predict(inputs)
        logging.info(str(res))
    except Exception as e:
        logging.info(e)
        return {}
    return res.tolist()
