import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_transformation import DataTransformation


if __name__== "__main__":
    transformer = DataTransformation()
    print("Starting Data Transformation process...")
    train_path, test_path, preprocesor_path = transformer.initiate_data_transformation()
    print(f"Transformed train data saved at: {train_path}")
    print(f"Transformed test data saved at: {test_path}")
    print(f"Preprocessor saved at: {preprocesor_path}")
    logging.info("Data Transformation process finished.")
    print("Data Transformation process finished.")
 