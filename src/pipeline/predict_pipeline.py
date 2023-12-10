import sys
import pandas as pd 
import pickle
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")  
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')  

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 geographic_region: str,
                 loyalty_status: str,
                 neighborhood_cluster: str,
                 affluence_grade: int,
                 age: int,
                 loyalty_card_tenure: int):
        self.gender = gender
        self.geographic_region = geographic_region
        self.loyalty_status = loyalty_status
        self.neighborhood_cluster = neighborhood_cluster
        self.affluence_grade = affluence_grade
        self.age = age
        self.loyalty_card_tenure = loyalty_card_tenure

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.gender],
                "Geographic Region": [self.geographic_region],
                "Loyalty Status": [self.loyalty_status],
                "Neighborhood Cluster-7 Level": [self.neighborhood_cluster],
                "Affluence Grade": [self.affluence_grade],
                "Age": [self.age],
                "Loyalty Card Tenure": [self.loyalty_card_tenure],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
