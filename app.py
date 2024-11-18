from flask import Flask, send_file, request, render_template
import sys, os, shutil
from src.exception import CustomException
from src.logger import logging
from src.pipeline.classification_model_train_pipeline import ClfTrainPipeline
from src.pipeline.regression_model_train_pipeline import RegTrainPipeline
from src.components import data_preprocessing
from src.pipeline import prediction_pipeline
from src.components import data_ingestion , deployment
import zipfile

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to my application"

@app.route("/predict_classification", methods=['POST', 'GET'])
def upload_classification():
    try:
        if request.method == "POST":
            data_ingestion.DataIngestion.initiate_data_ingestion(request)
            data_preprocessing.DataPreprocessor().transformer()

            logging.info("Executing the training pipeline for Classification Problem")
            train_pipeline = ClfTrainPipeline()
            train_pipeline.classification_training_pipeline()
            logging.info("Training Completed!")

            predict_pipeline = prediction_pipeline.ModelPrediction()
            prediction_file_detail = predict_pipeline.prediction()
            logging.info("Prediction Completed")

            logging.info("Classification prediction completed. Downloading prediction file and analysis.")

            # Define the path for the combined ZIP file
            combined_zip_path = os.path.join("artifacts", "combined_results.zip")

            # Add both files to the ZIP archive
            with zipfile.ZipFile(combined_zip_path, 'w') as zipf:
                zipf.write(prediction_file_detail.result_path, arcname=prediction_file_detail.prediction_file_name)
                zipf.write(os.path.join("artifacts", "data_analysis.zip"), arcname="analysis.zip")

            # Send the combined ZIP file
            response = send_file(combined_zip_path, download_name="combined_results.zip", as_attachment=True)
            
            # Clean up artifacts after the file is sent
            delete_artifacts()  
            return response
        else:
            return render_template('upload_classification.html')
    except Exception as e:
        logging.error(f"Error in upload_classification: {e}")
        return f"Error occurred: {e}", 500



@app.route("/predict_regression", methods=['POST', 'GET'])
def upload_regression():
    try:
        if request.method == "POST":
            data_ingestion.DataIngestion.initiate_data_ingestion(request)
            data_preprocessing.DataPreprocessor().transformer()

            logging.info("Executing the training pipeline for Regression Problem")
            train_pipeline = RegTrainPipeline()
            train_pipeline.regression_training_pipeline()
            logging.info("Training Completed!")

            predict_pipeline = prediction_pipeline.ModelPrediction()
            prediction_file_detail = predict_pipeline.prediction()
            logging.info("Prediction Completed")

            # Define the path for the combined ZIP file
            combined_zip_path = os.path.join("artifacts", "combined_results.zip")

            # Add both files to the ZIP archive
            with zipfile.ZipFile(combined_zip_path, 'w') as zipf:
                zipf.write(prediction_file_detail.result_path, arcname=prediction_file_detail.prediction_file_name)
                zipf.write(os.path.join("artifacts", "data_analysis.zip"), arcname="analysis.zip")

            # Send the combined ZIP file
            response = send_file(combined_zip_path, download_name="combined_results.zip", as_attachment=True)
            
            # Clean up artifacts after the file is sent
            delete_artifacts()  
            return response
        else:
            return render_template('upload_regression.html')

    except Exception as e:
        raise CustomException(e, sys)

@app.route("/deploy", methods=['POST', 'GET'])
def deploy_model():
        try:
            if request.method == "POST":
                if request=='AWS':
                    deployment.DeployModel.deploy_on_aws()
                else:
                    deployment.DeployModel.deploy_on_gcp()
            else:
                return render_template('deployment.html')

        except Exception as e:
            raise CustomException(e, sys)


def delete_artifacts():
    folder_path = "artifacts"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Skip the "model.pkl" file
            if filename == "model.pkl":
                logging.info(f"Skipped deletion of {file_path}")
                continue

            # Delete other files and directories
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
                logging.info(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory
                logging.info(f"Deleted directory: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete {file_path}. Reason: {e}")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
