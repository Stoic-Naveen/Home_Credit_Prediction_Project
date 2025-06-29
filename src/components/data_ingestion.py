import os
import pandas as pd
import sys
from src.constant import APPLICATION_TRAIN_PATH, APPLICATION_TEST_PATH, artifacts_folder
from dataclasses import dataclass # removes the need to write __init__
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    artifacts_folder: str = "artifacts"
    train_file_name: str = "application_train.csv"
    test_file_name: str = "application_test.csv"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig() # self.config because we are getting the data from some other folder

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            os.makedirs(self.config.artifacts_folder, exist_ok=True)
            logging.info(f"Artifacts folder created at: {self.config.artifacts_folder}")
            dst_path = os.path.join(self.config.artifacts_folder, self.config.train_file_name)
            logging.info(f"Copying data from {APPLICATION_TRAIN_PATH} to {dst_path}")
            df=pd.read_csv(APPLICATION_TRAIN_PATH)
            df.to_csv(dst_path, index=False)
            logging.info(f"Data saved to {dst_path}")
            logging.info("data Ingestion completed successfully")
        
        except Exception as e:  
            logging.error(f"Error occured during data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__== "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    logging.info("Data Ingestion process finished.")