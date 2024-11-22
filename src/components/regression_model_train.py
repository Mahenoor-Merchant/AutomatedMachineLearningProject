import sys
import joblib  
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from math import sqrt
import os
from sklearn.metrics import mean_squared_error
from src.utils import save_object
from dataclasses import dataclass
from docx import Document 

@dataclass
class RegModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    X_train_path: str = os.path.join("artifacts","preprocessed_files","X_train.csv")
    X_valid_path: str = os.path.join("artifacts","preprocessed_files","X_valid.csv")
    y_train_path: str = os.path.join("artifacts","preprocessed_files","y_train.csv")
    y_valid_path: str = os.path.join("artifacts","preprocessed_files","y_valid.csv")
    word_doc_path = os.path.join("artifacts", "data_analysis.docx")

class RegressionModelTrainer:
    def __init__(self):
        
        self.model_trainer_config = RegModelTrainerConfig()
        self.X_train=pd.read_csv(self.model_trainer_config.X_train_path)
        self.X_valid=pd.read_csv(self.model_trainer_config.X_valid_path)
        self.y_train=pd.read_csv(self.model_trainer_config.y_train_path).squeeze()
        self.y_valid=pd.read_csv(self.model_trainer_config.y_valid_path).squeeze()

    def BestModelSelector(self):
        regression_models = [
            LinearRegression,
            Lasso,
            Ridge,
            ElasticNet,
            SVR,
            DecisionTreeRegressor,
            RandomForestRegressor,
            GradientBoostingRegressor,
            AdaBoostRegressor,
            XGBRegressor,
            KNeighborsRegressor,
            BayesianRidge
        ]

        logging.info("Trying different models")
        model_rmse = {}
        best_model = None
        best_rmse = float('inf')  # Initialize with a high value for comparison

        for model in regression_models:
            try:
                regressor = model()  # Instantiate the regression model
                regressor.fit(self.X_train, self.y_train)  # Train the model
                y_pred_valid = regressor.predict(self.X_valid)  # Predict on validation data
                rmse = sqrt(mean_squared_error(self.y_valid, y_pred_valid))  # Calculate RMSE
                model_rmse[model.__name__] = rmse  # Store RMSE for the model

                logging.info(f"{model.__name__} completed with RMSE: {rmse}")

                # Update the best model if current model has a lower RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = regressor

            except Exception as e:
                logging.error(f"Error training {model.__name__}: {e}")  
                continue  # Skip to the next model if an error occurs

        if best_model:
            logging.info(f"Best model: {best_model.__class__.__name__} with RMSE: {best_rmse}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Write best model info to Word document
            doc = Document(self.model_trainer_config.word_doc_path)
            doc.add_paragraph(f"Best model for provided data is: {best_model.__class__.__name__} with RMSE: {best_rmse}. This model has been downloaded.")
            doc.save(self.model_trainer_config.word_doc_path)

            return "Model saved"
        else:
            logging.error("No valid models were successfully trained.")
            raise CustomException("All models failed to train.", sys)
