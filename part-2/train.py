"""
This script defines a model, trains it on a dataset, and saves the model to a file.
"""
import pickle

from sklearn.ensemble import RandomForestClassifier

# Define the model.
clf = RandomForestClassifier(random_state=0)

# Define the training dataset.
X = [[ 1,  2,  3], [11, 12, 13]]
y = [0, 1]

# Train the model on the dataset.
clf.fit(X, y)

# Save the model as a pickle file.
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
