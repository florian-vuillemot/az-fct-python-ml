import json

import azure.functions as func
from sklearn.ensemble import RandomForestClassifier

# Create the Azure Function App.
# The level ANONYMOUS indicate there is no authentification need to access this API.
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Defining the API endpoint.
# This route listens `/predict`.
@app.route(route="predict")
def predict(req: func.HttpRequest):
    # Retrieving the data to predict.
    # For education purposes, there is no input validation.
    to_predict = req.get_json()

    # Defining the model.
    clf = RandomForestClassifier(random_state=0)

    # Defining the training dataset.
    X = [[ 1,  2,  3],
        [11, 12, 13]]
    y = [0, 1]

    # Training the model on the dataset.
    clf.fit(X, y)

    # Do the prediction.
    prediction = clf.predict(to_predict)

    # Converting the prediction for HTTP output.
    res = json.dumps(prediction.tolist())

    # Return the HTTP result.
    return func.HttpResponse(res, status_code=200)
