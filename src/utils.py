import os
import sys

import boto3
import dill
import numpy as np
import pandas as pd
import yaml
from src.exception import CustomException



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def download_model(bucket_name, bucket_file_name, dest_file_name):
    try:
        s3_client = boto3.client("s3")

        s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)

        return dest_file_name

    except Exception as e:
        raise CustomException(e, sys)



class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys)

