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
from docx import Document 

@dataclass
class ClsModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    X_train_path: str = os.path.join("artifacts","preprocessed_files","X_train.csv")
    X_valid_path: str = os.path.join("artifacts","preprocessed_files","X_valid.csv")
    y_train_path: str = os.path.join("artifacts","preprocessed_files","y_train.csv")
    y_valid_path: str = os.path.join("artifacts","preprocessed_files","y_valid.csv")
    word_doc_path = os.path.join("artifacts", "data_analysis.docx")


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
            CatBoostClassifier,
            GaussianNB,
            MultinomialNB,
            BernoulliNB
        ]

        logging.info("Model accuracy check started")

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

                logging.info(f"{model.__name__} : {accuracy}")
                # Check if this model has the highest accuracy so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = classifier  # Save the best model object

                logging.info(f"{model.__name__} completed with accuracy: {accuracy}")

            except Exception as e:
                logging.error(f"Error training {model.__name__}: {e}")
                continue  # Move to the next model if there's an error

        if best_model:
            logging.info(f"Best model: {best_model} with accuracy: {best_accuracy}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Writing best model info to Word document
            doc = Document(self.model_trainer_config.word_doc_path)
            doc.add_paragraph(f"Best model for provided data is: {best_model.__class__.__name__} with accuracy: {best_accuracy}. This model has been downloaded.")
            doc.save(self.model_trainer_config.word_doc_path)

            return "Model saved"
        else:
            logging.error("No valid models were successfully trained.")
            raise CustomException("All models failed to train.", sys)
