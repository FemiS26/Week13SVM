import pandas as pd
import joblib

model = joblib.load("svm_model.sav")
feature_cols = joblib.load("svm_model.pkl")

def prepare_input(row):
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, dtype=int)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    return df

def predict(df):
    return model.predict(df)[0]
