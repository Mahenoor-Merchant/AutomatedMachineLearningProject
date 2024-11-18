from src.logger import logging
from src.exception import CustomException
import sys
import boto3
import os
from google.cloud import storage

class DeployModel:
    def __init__(self):
        self.model = os.path.join("artifacts", "model.pkl")
        self.s3_bucket_name = "your-s3-bucket-name"  # AWS S3 bucket name
        self.gcp_bucket_name = "your-gcp-bucket-name"  # GCP bucket name

    def deploy_on_aws(self):
        try:
            logging.info("Deploying model on AWS S3")
            s3 = boto3.client('s3')
            # Upload the model to AWS S3
            s3.upload_file(self.model, self.s3_bucket_name, os.path.basename(self.model))
            logging.info(f"Model successfully deployed to AWS S3 bucket {self.s3_bucket_name}")
            return "Model successfully deployed on AWS S3"
        except Exception as e:
            raise CustomException(f"Error deploying on AWS: {str(e)}", sys)
        
    def deploy_on_gcp(self):
        try:
            logging.info("Deploying model on GCP Storage")
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket_name)
            blob = bucket.blob(os.path.basename(self.model))
            # Upload the model to GCP Storage
            blob.upload_from_filename(self.model)
            logging.info(f"Model successfully deployed to GCP bucket {self.gcp_bucket_name}")
            return "Model successfully deployed on GCP Storage"
        except Exception as e:
            raise CustomException(f"Error deploying on GCP: {str(e)}", sys)