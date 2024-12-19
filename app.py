from flask import Flask, send_file, request, render_template
import sys, os, shutil, zipfile
from src.exception import CustomException
from src.logger import logging
from src.pipeline.classification_model_train_pipeline import ClfTrainPipeline
from src.pipeline.regression_model_train_pipeline import RegTrainPipeline
from src.components import data_preprocessing
from src.pipeline import prediction_pipeline
from src.components import data_ingestion
from src.components.deployment   import DeployModel  # Correct import


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

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

            
            word_doc_path = os.path.join("artifacts", "data_analysis.docx")
            if not os.path.exists(word_doc_path):
                logging.error("Analysis Word document not found.")
                return "Analysis Word document could not be generated.", 500
            
            modelpkl_path=os.path.join("artifacts","model.pkl")

            zip_filename = "output_files.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                zipf.write(word_doc_path, "data_analysis.docx")
                zipf.write(prediction_file_detail.result_path, prediction_file_detail.prediction_file_name)
                zipf.write(modelpkl_path,"model.pkl")

            # Send the Word document
            response = send_file('output_files.zip')

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
            # Step 1: Data ingestion and preprocessing
            data_ingestion.DataIngestion.initiate_data_ingestion(request)
            data_preprocessing.DataPreprocessor().transformer()

            # Step 2: Training pipeline
            logging.info("Executing the training pipeline for Regression Problem")
            train_pipeline = RegTrainPipeline()
            train_pipeline.regression_training_pipeline()
            logging.info("Training Completed!")

            # Step 3: Prediction pipeline
            predict_pipeline = prediction_pipeline.ModelPrediction()
            prediction_file_detail = predict_pipeline.prediction()
            logging.info("Prediction Completed")

            # Step 4: Word document for analysis
            word_doc_path = os.path.join("artifacts", "data_analysis.docx")
            if not os.path.exists(word_doc_path):
                logging.error("Analysis Word document not found.")
                return "Analysis Word document could not be generated.", 500
            
            modelpkl_path=os.path.join("artifacts","model.pkl")

            # Send the Word document
            zip_filename = "output_files.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                zipf.write(word_doc_path, "data_analysis.docx")
                zipf.write(prediction_file_detail.result_path, prediction_file_detail.prediction_file_name)
                zipf.write(modelpkl_path,"model.pkl")

            # Send the Word document
            response = send_file('output_files.zip')

            # Clean up artifacts after the file is sent
            delete_artifacts()
            return response
        else:
            return render_template('upload_regression.html')
    except Exception as e:
        logging.error(f"Error in upload_regression: {e}")
        return f"Error occurred: {e}", 500


@app.route("/deploy", methods=['POST', 'GET'])  
def deploy_model():
    try:
        if request.method == "POST":
            platform = request.form.get("deployment")  # Get selected platform (AWS/GCP)
            deployer = DeployModel()  # Create an instance of the DeployModel class

            # Perform the deployment based on selected platform
            if platform == "AWS":
                result= deployer.deploy_on_aws()  # Call the method on the instance
            elif platform == "GCP":
                result= deployer.deploy_on_gcp()  # Call the method on the instance

            return render_template('deployment_result.html', message=result, error=False)
            
        else:
            return render_template('deployment.html')
    except Exception as e:
        logging.error(f"Error in deploy_model: {e}")
        return f"Error occurred: {e}", 500


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
