import pandas as pd
import joblib

FEATURE_LIST = [
    'surface_thermal_radiation_downwards_sum', 'skin_temperature',
    'soil_temperature_level_2', 'surface_net_thermal_radiation_sum',
    'leaf_area_index_high_vegetation', 'skin_reservoir_content',
    'soil_temperature_level_1', 'dewpoint_temperature_2m',
    'soil_temperature_level_4', 'soil_temperature_level_3',
    'v_component_of_wind_10m', 'surface_net_solar_radiation_sum',
    'surface_solar_radiation_downwards_sum', 'u_component_of_wind_10m',
    'surface_sensible_heat_flux_sum'
]

class GHI_MODEL:
    def __init__(self, model_weight='../weights/RandomForest.joblib'):
        self.model = self.load_weight(model_weight)
    
    def load_weight(self, model_weight):
        model = joblib.load(model_weight)
        return model

    def predict(self, data_dict):
        if not check_features(data_dict):
            raise Exception(f"data_dict does not contain all required data")
        if isinstance(data_dict, str):
            features = [pd.DataFrame(data_dict)]
        elif isinstance(data_dict, list):
            features = pd.DataFrame(data_dict)
        else:
            raise Exception(f"data_dict isn't a list or string")
        response = self.model.predict(features)
        return response

def check_features(data_dict, feature_list=FEATURE_LIST):
    features = list(data_dict.keys())
    return len(features and feature_list) == len(feature_list)