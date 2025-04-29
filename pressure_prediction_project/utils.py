# utils.py
import numpy as np
import pandas as pd
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn_features.transformers import DataFrameSelector

# Load dataset
FILE_PATH = os.path.join(os.getcwd(), 'pressure.csv')
pressure = pd.read_csv(FILE_PATH)
X = pressure.drop(columns=['Blood_Pressure_Abnormality'], axis=1)
y = pressure['Blood_Pressure_Abnormality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)

# Preprocessing pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
X_train_num = num_pipeline.fit_transform(X_train)

# Load the trained model
model = joblib.load("pressure_model.sav.pkl")

# Feature order (used for prediction)
FEATURE_ORDER = [
    'Level_of_Hemoglobin', 'Age', 'BMI', 'Gender', 'Smoking',
    'Physical_activity', 'salt_content_in_the_diet', 
    'Chronic_kidney_disease', 'Adrenal_and_thyroid_disorders'
]

# User input encoder
def encode_user_input(user_data: dict) -> pd.DataFrame:
    encoding_maps = {
        'Gender': {'male': 0, 'female': 1},
        'Smoking': {'yes': 1, 'no': 0},
        'Chronic_kidney_disease': {'yes': 1, 'no': 0},
        'Adrenal_and_thyroid_disorders': {'yes': 1, 'no': 0}
    }
    
    encoded_data = {}
    for key, value in user_data.items():
        if key in encoding_maps:
            encoded_data[key] = encoding_maps[key].get(value.lower(), -1)
        else:
            encoded_data[key] = float(value)

    return pd.DataFrame([encoded_data], columns=FEATURE_ORDER)

# Preprocess new input
def preprocess_new(X_new):  
    return num_pipeline.transform(X_new)

# Final prediction function
def predict_pressure(user_input: dict) -> int:
    encoded_df = encode_user_input(user_input)
    processed = preprocess_new(encoded_df)
    prediction = model.predict(processed)
    return int(prediction[0])
