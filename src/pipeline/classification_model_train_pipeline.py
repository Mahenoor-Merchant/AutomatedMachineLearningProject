import sys
from src.exception import CustomException
from src.logger import logging
from src.components import data_analysis
from src.components import classification_model_train


class ClfTrainPipeline:
    def __init__(self)-> None:

        logging.info('Training Pipeline Initiated')
        self.data_analyzer=data_analysis.DataAnalysis()
        self.model_trainer=classification_model_train.ClassificationModelTrainer()

    def classification_training_pipeline(self):
        try:

            logging.info("Creating Plots")
            self.data_analyzer.analyze_post()

            logging.info("Training and Selecting Best Model")
            self.model_trainer.BestModelSelector()
            
            return "TRAINING PIPELINE COMPLETED"
        
        except Exception as e:
            raise CustomException(e,sys)