"""
This script defines a model, trains it on a dataset, and saves the model to a file.
"""
import sys
import pickle

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define the model.
clf = RandomForestClassifier(random_state=0)

# Define the training dataset.
X, y = make_classification(
    n_samples=1000, n_features=3,
    n_informative=2, n_redundant=0,
    random_state=0, shuffle=False
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model on the dataset.
clf.fit(X_train, y_train)

# Save the model as a pickle file.
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Evaluate the model.
scores = cross_val_score(clf, X_test, y_test, cv=5)

# Print the accuracy.
msg = f"### {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}"
if scores.mean() > 0.9:
    print(f'{msg} :rocket:')
    # Exit with a success code.
    sys.exit(0)
else:
    print(f'{msg} :no_entry:')
    # Exit with a failure code.
    sys.exit(1)
