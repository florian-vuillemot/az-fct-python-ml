import json

import azure.functions as func
from sklearn.ensemble import RandomForestClassifier

# Create the Azure Function App.
# The ANONYMOUS level indicates no authentification is needed to access this API.
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Defining the API endpoint.
# This route listens `/predict`.
@app.route(route="predict")
def predict(req: func.HttpRequest):
    # Retrieve the data to be predicted.
    # For education purposes, there is no input validation.
    to_predict = req.get_json()

    # Define the model.
    clf = RandomForestClassifier(random_state=0)

    # Define the training dataset.
    X = [[ 1,  2,  3], [11, 12, 13]]
    y = [0, 1]

    # Train the model on the dataset.
    clf.fit(X, y)

    # Run the prediction.
    prediction = clf.predict(to_predict)

    # Convert the prediction for HTTP output.
    res = json.dumps(prediction.tolist())

    # Return the HTTP result.
    return func.HttpResponse(res, status_code=200)
