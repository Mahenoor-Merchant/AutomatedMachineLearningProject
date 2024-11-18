import os, sys
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    input_csv_file_path: str = os.path.join("artifacts", "input_csv_file.csv")

    X_data_file_path=os.path.join("artifacts","X_data.csv")
    y_data_file_path=os.path.join("artifacts","y_data.csv")

    numerical_cols_path=os.path.join("artifacts", "numerical_cols.csv")
    categorical_cols_path=os.path.join("artifacts", "categorical_cols.csv")
    

    X_train_path: str = os.path.join("artifacts","preprocessed_files","X_train.csv")
    X_test_path: str = os.path.join("artifacts","preprocessed_files","X_test.csv")
    X_valid_path: str = os.path.join("artifacts","preprocessed_files","X_valid.csv")
    y_train_path: str = os.path.join("artifacts","preprocessed_files","y_train.csv")
    y_test_path: str = os.path.join("artifacts","preprocessed_files","y_test.csv")
    y_valid_path: str = os.path.join("artifacts","preprocessed_files","y_valid.csv")
class DataPreprocessor:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def NumCat(self):
        try:
            logging.info("separating X and y_data")
            df = pd.read_csv(self.data_transformation_config.input_csv_file_path)
            if 'id' in df.columns:
                df = df.drop("id", axis=1)
            if 'name' in df.columns:
                df = df.drop("name", axis=1)
            if 'Id' in df.columns:
                df=df.drop("ID",axis=1)
                
            df=df.drop_duplicates()

            target_column = 'target'
            y_data = df[target_column]
            X_data = df.drop(columns=[target_column])

            # Save X and y data
            X_data.to_csv(self.data_transformation_config.X_data_file_path, index=False)
            y_data.to_csv(self.data_transformation_config.y_data_file_path, index=False)

            # Identify numerical and categorical columns
            logging.info("Separating numerical and categorical columns")
            numerical_cols = X_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X_data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

            # Save numerical columns only if they exist
            if numerical_cols:
                logging.info("Saving numerical columns")
                X_data[numerical_cols].to_csv(self.data_transformation_config.numerical_cols_path, index=False)

            # Save categorical columns only if they exist
            if categorical_cols:
                logging.info("Saving categorical columns")
                X_data[categorical_cols].to_csv(self.data_transformation_config.categorical_cols_path, index=False)



            return numerical_cols, categorical_cols
        except Exception as e:
            raise CustomException(e, sys)

    def numerical_preprocessing(self):
        try:
            logging.info("Entered numerical preprocessing pipeline")
            numerical_preprocessor = Pipeline(
                steps=[
                    ("imputation", SimpleImputer(strategy="median")),
                    ("standard_scaling", StandardScaler())
                ]
            )    
            return numerical_preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def categorical_preprocessing(self):
        try:
            logging.info("Entered categorical preprocessing pipeline")
            categorical_preprocessor = Pipeline(
                steps=[
                    ("imputation", SimpleImputer(strategy="most_frequent")),
                    ("Encoding", OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            return categorical_preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def column_transforming(self):
        try:
            logging.info("Entered column transformer")
            # Get column lists
            numerical_cols, categorical_cols = self.NumCat()
            
            # Instantiate preprocessor pipelines
            numerical_preprocessor = self.numerical_preprocessing()
            categorical_preprocessor = self.categorical_preprocessing()
            
            # Define the ColumnTransformer
            column_transformer = ColumnTransformer(
                transformers=[
                    ("categorical", categorical_preprocessor, categorical_cols),
                    ("numerical", numerical_preprocessor, numerical_cols)
                ]
            )
            return column_transformer
        except Exception as e:
            raise CustomException(e, sys)

    def transformer(self):
        try:
            """self.NumCat()"""
            logging.info("Transforming data")
            # Prepare column transformer
            column_transformer = self.column_transforming()
            X_data = pd.read_csv(self.data_transformation_config.X_data_file_path)
            y_data = pd.read_csv(self.data_transformation_config.y_data_file_path).squeeze()  # Convert to Series

            transformed_data = column_transformer.fit_transform(X_data)
           
            # Target feature transformation if categorical
            if y_data.dtype == 'object' or y_data.dtype.name == 'category':
                label_encoder = LabelEncoder()
                transformed_target = label_encoder.fit_transform(y_data)
                transformed_target = transformed_target.ravel()  
            else:
                transformed_target = y_data.values
                # Assume X and y are already defined
                transformed_target = transformed_target.ravel()  


            # Split data
            logging.info("Performing splits")
            X_train, X_test, y_train, y_test = train_test_split(transformed_data, transformed_target, test_size=0.30, random_state=42)
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

            # Save split data to artifacts directory
            os.makedirs("artifacts/preprocessed_files", exist_ok=True)
            pd.DataFrame(X_train).to_csv(self.data_transformation_config.X_train_path, index=False)
            pd.DataFrame(X_test).to_csv(self.data_transformation_config.X_test_path, index=False)
            pd.DataFrame(X_valid).to_csv(self.data_transformation_config.X_valid_path, index=False)
            pd.DataFrame(y_train.ravel()).to_csv(self.data_transformation_config.y_train_path, index=False)
            pd.DataFrame(y_test.ravel()).to_csv(self.data_transformation_config.y_test_path, index=False)
            pd.DataFrame(y_valid.ravel()).to_csv(self.data_transformation_config.y_valid_path, index=False)

            logging.info("Data preprocessing completed")

            return (
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
