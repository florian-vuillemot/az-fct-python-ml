"""
This script defines a model, trains it on a dataset, and saves the model to a file.
"""
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Define the model.
clf = RandomForestClassifier(random_state=0)

# Define the training dataset.
X, y = make_classification(
    n_samples=1000, n_features=2,
    n_informative=2, n_redundant=0,
    random_state=0, shuffle=False
)

# Train the model on the dataset.
clf.fit(X, y)

# Save the model as a pickle file.
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Evaluate the model.
scores = cross_val_score(clf, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))