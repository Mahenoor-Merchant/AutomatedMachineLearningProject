from src.exception import CustomException
import os, sys
from src.logger import logging

class DataIngestion:
    
    def __init__(self):
        self.input_csv_file_path: str = os.path.join("artifacts", "input_csv_file.csv")

    def initiate_data_ingestion(request) -> str:
        try:
            logging.info("Making directory for data ingestion")
            pred_file_input_dir = "artifacts"
            if not os.path.exists(pred_file_input_dir):
                os.makedirs(pred_file_input_dir)
            
            logging.info("Accessing files")

            # Access the uploaded file
            input_csv_file = request.files["file"]
            # Save the file

            logging.info("saving the file")
            input_csv_file.save(r"artifacts\input_csv_file.csv")

            return "Data ingested sucessfully"
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys)