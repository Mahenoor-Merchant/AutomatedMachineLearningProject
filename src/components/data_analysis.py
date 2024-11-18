import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from src.components import data_preprocessing
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from flask import Flask, send_file
from io import StringIO
import zipfile
# Ensure Agg backend is used for matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@dataclass
class AnalysisConfig:
    numerical_analysis_file_path = os.path.join("artifacts", "numerical.jpg")
    categorical_analysis_file_path = os.path.join("artifacts", "categorical.jpg")
    heatmap_file_path = os.path.join("artifacts", "heatmap.jpg")
    descriptive_file_path = os.path.join("artifacts", "descriptive.txt")
    zip_file_path = os.path.join("artifacts", "data_analysis.zip")

class DataAnalysis:
    def __init__(self):
        self.data_transformation_config = data_preprocessing.DataTransformationConfig()
        self.analysis_config = AnalysisConfig()
        
        # Ensure the directory exists for saving files
        if not os.path.exists('artifacts'):
            os.makedirs('artifacts')

        self.numerical_cols, self.categorical_cols = data_preprocessing.DataPreprocessor().NumCat()
        self.df_path = os.path.join("artifacts", "input_csv_file.csv")
        self.df = pd.read_csv(self.df_path)

        # Load numerical and categorical columns if the files exist
        if os.path.exists(self.data_transformation_config.numerical_cols_path):
            logging.info("Loading numerical columns from CSV")
            self.numerical_df = pd.read_csv(self.data_transformation_config.numerical_cols_path)
        else:
            logging.info("No numerical columns file found; setting numerical_df to None")
            self.numerical_df = pd.DataFrame()  # Use an empty DataFrame

        if os.path.exists(self.data_transformation_config.categorical_cols_path):
            logging.info("Loading categorical columns from CSV")
            self.categorical_df = pd.read_csv(self.data_transformation_config.categorical_cols_path)
        else:
            logging.info("No categorical columns file found; setting categorical_df to None")
            self.categorical_df = pd.DataFrame()  # Use an empty DataFrame

    def create_numerical_plots(self):
        try:
            logging.info("Creating numerical plots")    
            plt.figure(figsize=(15, 15))
            plt.suptitle("Universal Analysis of Numerical Features", fontsize=20)

            num_rows = len(self.numerical_cols) // 3 + (len(self.numerical_cols) % 3 > 0)
            for i in range(len(self.numerical_cols)):
                plt.subplot(num_rows, 3, i + 1)
                sns.kdeplot(x=self.numerical_df[self.numerical_cols[i]], fill=True, color='r')  
                plt.grid(True)
                plt.xlabel(self.numerical_cols[i])

            plt.tight_layout()
            plt.savefig(self.analysis_config.numerical_analysis_file_path)
            plt.close()

            logging.info("Creating heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.numerical_df.corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.savefig(self.analysis_config.heatmap_file_path)
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def create_categorical_plots(self):
        try:
            logging.info("Creating categorical plots")
            plt.figure(figsize=(15, 15))
            plt.suptitle("Universal Analysis of Categorical Features", fontsize=20)

            n_cols = 2
            n_rows = len(self.categorical_cols) // n_cols + (len(self.categorical_cols) % n_cols > 0)

            for i in range(len(self.categorical_cols)):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.countplot(x=self.categorical_df[self.categorical_cols[i]], palette='Set2')
                plt.xlabel(self.categorical_cols[i])
                plt.xticks(rotation=45)
                plt.grid(True)

            plt.tight_layout()
            logging.info("Saving categorical plot")
            plt.savefig(self.analysis_config.categorical_analysis_file_path)
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def descriptive_analysis(self):
        try:
            logging.info("Performing descriptive analysis")
            desc = self.df.describe()

            with open(self.analysis_config.descriptive_file_path, 'w') as f:
                f.write("Descriptive Statistics:\n")
                f.write(desc.to_string())

            buffer = StringIO()
            self.df.info(buf=buffer)
            info = buffer.getvalue()

            with open(self.analysis_config.descriptive_file_path, 'a') as f:
                f.write("\nDataFrame Info:\n")
                f.write(info)

            logging.info("Descriptive analysis saved successfully")
            
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            raise CustomException(e, sys)

    def create_zip_file(self):
        try:
            logging.info("Creating zip file of all analysis results")

            # Create a Zip file
            with zipfile.ZipFile(self.analysis_config.zip_file_path, 'w') as zipf:
                zipf.write(self.analysis_config.numerical_analysis_file_path, os.path.basename(self.analysis_config.numerical_analysis_file_path))
                zipf.write(self.analysis_config.categorical_analysis_file_path, os.path.basename(self.analysis_config.categorical_analysis_file_path))
                zipf.write(self.analysis_config.heatmap_file_path, os.path.basename(self.analysis_config.heatmap_file_path))
                zipf.write(self.analysis_config.descriptive_file_path, os.path.basename(self.analysis_config.descriptive_file_path))

            logging.info(f"Zip file created: {self.analysis_config.zip_file_path}")
        except Exception as e:
            logging.error(f"Error during zip file creation: {e}")
            raise CustomException(e, sys)
     
    
    def analyze_post(self):
        try:
            self.create_numerical_plots()
            self.create_categorical_plots()
            self.descriptive_analysis()
            self.profilling()
            self.create_zip_file()  # Create a zip file containing all analysis results
            logging.info("Data analysis is completed.")
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            raise CustomException(e, sys)

