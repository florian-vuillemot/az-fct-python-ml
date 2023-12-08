"""
This script defines a model, trains it on a dataset, and saves the model to a file.
"""
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

# Evaluate the model.
scores = cross_val_score(clf, X, y)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))