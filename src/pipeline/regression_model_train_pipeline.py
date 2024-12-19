import sys, os
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import make_pipeline
from src.components import data_ingestion
from src.components import data_preprocessing
from src.components import data_analysis
from src.components import regression_model_train

class RegTrainPipeline:
    def __init__(self)-> None:

        logging.info('Training Pipeline Initiated')
        self.data_analyzer=data_analysis.DataAnalysis()
        self.model_trainer=regression_model_train.RegressionModelTrainer()

    def regression_training_pipeline(self):
        try:
            
            logging.info("Creating Plots")
            self.data_analyzer.analyze_post()

            logging.info("Training and selecting the Best Model")
            
            self.model_trainer.BestModelSelector()
            return "Training pipeline completed"
        
        except Exception as e:
            raise CustomException(e,sys)