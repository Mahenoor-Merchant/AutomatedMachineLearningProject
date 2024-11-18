import os
import sys
import joblib  # Import joblib for saving models
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class ClsModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    X_train_path: str = os.path.join("artifacts","preprocessed_files","X_train.csv")
    X_valid_path: str = os.path.join("artifacts","preprocessed_files","X_valid.csv")
    y_train_path: str = os.path.join("artifacts","preprocessed_files","y_train.csv")
    y_valid_path: str = os.path.join("artifacts","preprocessed_files","y_valid.csv")

class ClassificationModelTrainer:
    def __init__(self):
        self.model_trainer_config = ClsModelTrainerConfig()
        self.X_train=pd.read_csv(self.model_trainer_config.X_train_path)
        self.X_valid=pd.read_csv(self.model_trainer_config.X_valid_path)
        self.y_train=pd.read_csv(self.model_trainer_config.y_train_path).squeeze()
        self.y_valid=pd.read_csv(self.model_trainer_config.y_valid_path).squeeze()

    def BestModelSelector(self):
        classification_models = [
            LogisticRegression,
            SVC,
            DecisionTreeClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier,
            AdaBoostClassifier,
            KNeighborsClassifier,
            XGBClassifier,
            LGBMClassifier,
            CatBoostClassifier,
            GaussianNB,
            MultinomialNB,
            BernoulliNB
        ]

        model_accuracy = {}
        best_model = None
        best_accuracy = 0

        logging.info("Trying different models!")
        for model in classification_models:
            try:
                classifier = model()  # Instantiate model
                classifier.fit(self.X_train, self.y_train)
                y_pred_valid = classifier.predict(self.X_valid)
                accuracy = accuracy_score(self.y_valid, y_pred_valid)
                model_accuracy[model.__name__] = accuracy  # Store model name and accuracy
                
                # Check if this model has the highest accuracy so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = classifier  # Save the best model object

                logging.info("Best Model saved! with accuracy: {best_accuracy}")
                return save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model,
                )
            except Exception as e:
                logging.error(f"Error training {model.__name__}: {e}")
                raise CustomException(e,sys)
