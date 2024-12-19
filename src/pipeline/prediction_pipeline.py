import sys, os
from src.exception import CustomException
from src.logger import logging
import joblib
import pandas as pd
from src.utils import MainUtils
from dataclasses import dataclass
from flask import send_file

@dataclass
class PredcitionConfig:
    X_test_path: str = os.path.join("artifacts","preprocessed_files","X_test.csv")
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    prediction_file_name:str =  "predicted_file.csv"
    prediction_output_dirname: str = "predictions"
    result_path=os.path.join("artifacts",prediction_output_dirname,prediction_file_name)

class ModelPrediction:
    def __init__(self):

        self.config=PredcitionConfig()
        logging.info("loading the model and test data")
        self.loaded_model = MainUtils.load_object(self.config.trained_model_file_path)
        self.X_test = pd.read_csv(self.config.X_test_path)

    def prediction(self):
        try:
            logging.info("Predicting results")
            y_pred = self.loaded_model.predict(self.X_test).squeeze()
            
            # Convert y_pred to a DataFrame
            y_pred_df = pd.DataFrame(y_pred, columns=["Prediction"])
            
            # Concatenate X_test and y_pred_df
            result = pd.concat([self.X_test, y_pred_df], axis=1)

            # Create directory if it does not exist
            os.makedirs(os.path.dirname(self.config.result_path), exist_ok=True)

            # Save the result to CSV
            result.to_csv(self.config.result_path, index=False)

            logging.info("Prediction completed")
            return self.config

        except Exception as e:
            raise CustomException(e, sys)
