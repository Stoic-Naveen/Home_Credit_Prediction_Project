import os
from dataclasses import dataclass
from src.utils.main_utils import MainUtils
import sys
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.constant import MODEL_FILE_NAME, MODEL_FILE_EXTENSION, artifacts_folder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join("artifacts")
    trained_model_path: str = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy: float = 0.45
    model_config_file_path: str = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            'XGBClassifier': XGBClassifier(n_jobs=-1, verbosity=1),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
            'GradientBoostingClassifier': GradientBoostingClassifier()
        }

        
        self.model_param_grid = self.utils.read_yaml_file(self.config.model_config_file_path)["model_selection"]["model"]
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        logging.info("Evaluating models....")
        report = {}
        for name, model in self.models.items():
            print(f"Training base model: {name}...")
            logging.info(f"Training base model: {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = score
            logging.info(f"{name} - Accuracy: {score:.4f}")
            print(f"Model evaluation report: {report}")
            return report

    def finetune_best_model(self, model_name, model, X_train, y_train):
        print(f"Starting GridSearchCV for {model_name}...")
        logging.info(f"Starting GridSearchCV for {model_name}...")
        param_grid = self.model_param_grid[model_name]["search_param_grid"]
        grid_search = GridSearchCV(model, param_grid = param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best parameters for {model_name}: {best_params}")
        logging.info(f"Best parameters for {model_name}: {best_params}")
        model.set_params(**best_params)
        return model
    
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        logging.info("Initiating model training...")
        try:
            # Evaluate base models
            logging.info("Evaluating base models...")
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test)
            best_model_name = max(model_report, key=model_report.get)
            best_model = self.models[best_model_name]
            logging.info(f"Best base model: {best_model_name} with accuracy: {model_report[best_model_name]}")

            if model_report[best_model_name] < self.config.expected_accuracy:
                raise CustomException(f"Best model accuracy {model_report[best_model_name]} is less than expected {self.config.expected_accuracy}")
            
            # Fine-tune the best model
            best_model = self.finetune_best_model(best_model_name, best_model, X_train, y_train)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            final_score = accuracy_score(y_test, y_pred)
            logging.info(f"Final {best_model_name} test accuracy after tuning: {final_score:.4f}")

            if final_score < self.config.expected_accuracy:
                raise CustomException(f"Final model accuracy {final_score} is less than expected {self.config.expected_accuracy}")

            # Save the best model
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok = True)

            self.utils.save_object(self.config.trained_model_path, best_model)
            
            
            logging.info(f"Best model saved at: {self.config.trained_model_path}")
            
            return self.config.trained_model_path
        
        
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys) from e