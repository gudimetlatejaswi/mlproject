import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
            params = {
                'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
                'Decision Tree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
                'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]},
                'Linear Regression': {},  
                'K-Neighbors Classifier': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
                'XGBClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
                'CatBoosting Classifier': {'iterations': [100, 200], 'learning_rate': [0.01, 0.1]},
                'AdaBoost Classifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            }

            model_report:dict=  evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test, models=models,param=params)
        
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ## To get best model name from dict
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both trining and testing dataset")
            
            save_object(
               file_path=self.model_trainer_config.trained_model_file_path,
               obj=best_model
            )
           
            predicted=best_model.predict(x_test)
            r2_square= r2_score(y_test, predicted)
            return r2_square
       
        except Exception as e:
            raise CustomException(e,sys)