# Serving a local model

Here we are serving a local machine learning model.

## How to run

Prerequisites: With a terminal, go in this foler (part-1).

First, install dependencies:
```
pip install -r requirements.txt
```

Then, build the model locally:
```
python fit.py
```
The file `model.pickle` is now locally generated.


Then, start the application:
```
flask --debug run
```

The application is now listenning on the port 5000 and available for HTTP request. If `curl` is installed, run:
```
$ curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '[[4, 5, 6], [14, 15, 16]]' http://localhost:5000/
[
  0,
  1
]
```


