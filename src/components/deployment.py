import os
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
import sys
import boto3
from google.cloud import storage

class DeployModel:
    def __init__(self):
        # Load environment variables from the .env file
        load_dotenv(os.path.join("src", "components", ".env"))
        
        # Retrieve the secrets from the environment
        self.model = os.path.join("artifacts", "model.pkl")
        self.s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")  # AWS S3 bucket name
        self.gcp_bucket_name = os.getenv("GCP_BUCKET_NAME")  # GCP bucket name
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")  # AWS Access Key
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")  # AWS Secret Key
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")  # GCP Project ID

    def deploy_on_aws(self):
        try:
            
            # Use AWS credentials from environment variables
            s3 = boto3.client('s3', aws_access_key_id=self.aws_access_key, aws_secret_access_key=self.aws_secret_key)
            
            # Upload the model to AWS S3
            s3.upload_file(self.model, self.s3_bucket_name, os.path.basename(self.model))
            logging.info(f"Model successfully deployed to AWS S3 bucket {self.s3_bucket_name}")
            return "Model successfully deployed on AWS S3"
        except Exception as e:
            raise CustomException(f"Error deploying on AWS: {str(e)}", sys)

    def deploy_on_gcp(self):
        try:
            logging.info("Deploying model on GCP Storage")
            # Use GCP credentials from environment variables
            storage_client = storage.Client(project=self.gcp_project_id)
            bucket = storage_client.bucket(self.gcp_bucket_name)
            blob = bucket.blob(os.path.basename(self.model))
            
            # Upload the model to GCP Storage
            blob.upload_from_filename(self.model)
            logging.info(f"Model successfully deployed to GCP bucket {self.gcp_bucket_name}")
            return "Model successfully deployed on GCP Storage"
        except Exception as e:
            raise CustomException(f"Error deploying on GCP: {str(e)}", sys)