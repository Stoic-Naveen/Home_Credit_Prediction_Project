import sys
import os
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import RobustScaler # Help in situation when we have outlier and we don't want to deal with outlier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.constant import APPLICATION_TEST_PATH, APPLICATION_TRAIN_PATH
from src.utils.main_utils import MainUtils



@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join("artifacts")
    ingested_train_path: str = os.path.join(artifact_dir, "application_train.csv")
    transformed_train_file_path: str = os.path.join(artifact_dir, "train.npy")
    transformed_test_file_path: str = os.path.join(artifact_dir, "test.npy")
    transformed_train_csv_path: str = os.path.join(artifact_dir, "transformed_train.csv")
    transformed_test_csv_path: str = os.path.join(artifact_dir, "transformed_test.csv")
    transformed_object_file_path: str = os.path.join(artifact_dir, "preprocessor.pkl")



class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.utils = MainUtils()
    
    def initiate_data_transformation(self):
        logging.info("Data Transformation started")
        try:
            df = pd.read_csv(self.config.ingested_train_path)
            logging.info(f"Data loaded from {self.config.ingested_train_path}")

            if 'SK_ID_CURR' in df.columns:
                df.drop(columns=["SK_ID_CURR"], inplace = True)
                logging.info("Dropped 'SK_ID_CURR' column from the dataset")

            X = df.drop(columns=['TARGET'], axis=1)
            y = df['TARGET']
            logging.info("Separatedd features and target variable")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
            logging.info("Train-Test split completed")
            logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            logging.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

            categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
            numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")


            # numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            logging.info("Numerical pipeline created with imputer and scaler")
            

            # fit and transform the training data to numerical pipeline
            X_train_num = numerical_pipeline.fit_transform(X_train[numerical_cols])
            logging.info("Training data numerical features transformed using the numerical pipeline")

            X_test_num = numerical_pipeline.transform(X_test[numerical_cols])
            logging.info("Test data numerical features transformed for the test set using the numerical pipeline")

            # Categorical pipeline
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
            logging.info("Categorical features transformed using one-hot encoding")

            # Align the categorical features in train and test sets
            X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
            logging.info("Aligned categorical features in train and test sets")

            # Combine numerical and categorical features
            X_train_processed = np.hstack((X_train_num, X_train_cat))
            X_test_processed = np.hstack((X_test_num, X_test_cat))
            logging.info("Combined numerical and categorical features for training and test sets")

            # Save the processed data
            np.save(self.config.transformed_train_file_path, {'X': X_train_processed, 'y': y_train.values})
            np.save(self.config.transformed_test_file_path, {'X': X_test_processed, 'y': y_test.values})
            logging.info(f"Transformed data saved to {self.config.transformed_train_file_path} and {self.config.transformed_test_file_path}")

            # Save the preprocessed data as CSV files
            all_feature_names = numerical_cols + list(X_train_cat.columns)
            train_df_out = pd.DataFrame(X_train_processed, columns=all_feature_names)
            test_df_out = pd.DataFrame(X_test_processed, columns=all_feature_names)
            train_df_out['TARGET'] = y_train.values
            test_df_out['TARGET'] = y_test.values

            test_df_out.to_csv(self.config.transformed_test_csv_path, index=False)
            train_df_out.to_csv(self.config.transformed_train_csv_path, index=False)
            logging.info(f"Transformed data saved as CSV files to {self.config.transformed_train_csv_path} and {self.config.transformed_test_csv_path}")

            # Save the preprocessor object
            preprocessor = {
                'numerical_pipeline': numerical_pipeline,
                'categorical_cols': categorical_cols,
                'categorical_cols': X_train_cat.columns.tolist(),
                'numerical_cols': numerical_cols
            }

            self.utils.save_object(self.config.transformed_object_file_path, preprocessor)
            logging.info(f"Preprocessor object saved to {self.config.transformed_object_file_path}")
            logging.info("Data Transformation completed successfully")
        
            return (
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path
            )
        
        
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys) from e