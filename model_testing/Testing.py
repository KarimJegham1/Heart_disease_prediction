#Unittest testing
import unittest
import numpy as np
from Model_training.Training import model,X_test
# Create a class called TestModelInference
class TestModelInference(unittest.TestCase):
	def setUp(self):
		self.model = model

		# set X_test as a class attribute
		self.X_test = X_test

	# define a test for prediction output values
	def test_prediction_output_values(self):
		print("Running test_prediction_output_values test case")

		# Get model predictions
		y_pred = self.model.predict(self.X_test)
		unique_values = np.unique(y_pred)
		for value in unique_values:
			self.assertIn(value, [0, 1])