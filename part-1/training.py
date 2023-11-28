"""
Simple training script saving the model locally.

This Python script trains a Random Forest Classifier model using Scikit-learn and then saves the trained model to a file for later use.
"""
import pickle

from sklearn.ensemble import RandomForestClassifier

# Defining my model.
clf = RandomForestClassifier(random_state=0)

# Defining my dataset.
X = [[ 1,  2,  3],
    [11, 12, 13]]
y = [0, 1]

# Training my model on the dataset.
clf.fit(X, y)

# Saving my model.
with open('model.pickle', 'wb') as pickle_file:
    # Converting the model to bytes.
    model_bytes = pickle.dumps(clf)
    # Writing the model bytes to a file.
    pickle_file.write(model_bytes)
