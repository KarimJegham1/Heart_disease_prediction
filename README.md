**Heart Disease Prediction Model**
This project is focused on developing a machine learning model to predict heart disease using a dataset of patient health metrics. The project involves preprocessing the data, training a model using different classifiers, performing feature selection, and logging predictions for analysis. Below is an overview of the main components of the project.

**Project Structure**
  **Model_training:**
**data_preprocessing.py**: This script is responsible for loading the dataset, cleaning the data by removing duplicates and handling missing values, and performing feature transformations
such as quantizing binary and multiclass columns. It normalizes the data and splits it into training and testing sets. It also saves the processed dataset to a CSV file.
**training.py**: This script trains a RandomForest classifier on the preprocessed dataset. 
It performs feature selection using SelectFromModel to identify the most important features. 
A Support Vector Classifier (SVC) is then trained on the transformed data, and the model is evaluated using cross-validation.
  **Feature_selection:**

**Features.py**: This script uses Feast, a feature store framework, to create and store features for the heart disease dataset.
It defines the relevant fields, including cp, thalach, ca, and thal, and stores the data in a Parquet file. The script creates a FeatureView and stores it in the local Feast feature store.
Feature Store:
Features such as cp, thalach, ca, and thal are selected and stored in Feast, a feature store platform, to facilitate feature management and tracking over time.
The features are stored in Parquet format, and the feature store is updated with the new data.
  **Logging:**

**logging.py**: This script logs the predictions made by the trained model on the test set. For each instance in the test set, it logs the predicted class along with the real class, providing insights into
the model's performance.Pedictions made by the trained model on the test set are logged with real and predicted class labels. This is useful for evaluating the model’s performance and for debugging.
**Conclusion**
This project demonstrates a full machine learning pipeline for predicting heart disease. It involves data preprocessing, feature selection, model training, and logging of predictions.
The model is trained using a RandomForest classifier, with feature selection performed to identify the most important features. 
Additionally, the project uses Feast for feature management, ensuring the features are tracked and stored properly for later use.

