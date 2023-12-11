"""
Sample test of the Azure Function application.
"""
import json
import pickle
import unittest

import azure.functions as func
from sklearn.ensemble import RandomForestClassifier

from function_app import predict


class TestFunction(unittest.TestCase):
    X = [[1, 2, 3], [11, 12, 13]]
    y = [0, 1]

    def setUp(self):
        """
        Create a model specifically for the test.
        """
        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X, self.y)
        with open('model.pkl', 'wb') as fd:
            fd.write(pickle.dumps(clf))

    def test_interface(self):
        # Construct a mock HTTP request.
        req = func.HttpRequest(
            method='POST',
            url='/api/predict',
            body=json.dumps(self.X).encode()
        )

        # Call the function.
        func_call = predict.build().get_user_function()
        resp = func_call(req)

        # Check the output.
        self.assertEqual(
            json.loads(resp.get_body()),
            self.y,
        )
        self.assertFalse(True)


if __name__ == "__main__":
    unittest.main()
