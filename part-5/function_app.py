"""
This file contains the Azure Function App.
It is a Python function that can be called over HTTP.
It loads the model and runs the prediction.
"""
import os
import json
import pickle

import azure.functions as func

# Create the Azure Function App.
# The ANONYMOUS level indicates no authentification is needed to access this API.
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Defining the API endpoint.
@app.route(route="predict")
def predict(req: func.HttpRequest):
    files_in_share = os.listdir("/path/to/mount")
    return func.HttpResponse(files_in_share, status_code=200)
    # Retrieve the data to be predicted.
    # For education purposes, there is no input validation.
    to_predict = req.get_json()

    # Load the model.
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Run the prediction.
    prediction = clf.predict(to_predict)

    # Convert the prediction for HTTP output.
    res = json.dumps(prediction.tolist())

    # Return the HTTP result.
    return func.HttpResponse(res, status_code=200)
