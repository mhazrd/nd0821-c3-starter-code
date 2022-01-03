# Put the code for your API here.

import pickle
import pandas as pd
import numpy as np

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI()

cols = 'age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country'.replace('-', '_').split(',')
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

class DataItem(BaseModel):
    age           : int
    fnlgt         : int
    education_num : int = Field(alias='education-num')
    capital_gain  : int = Field(alias='capital-gain')
    capital_loss  : int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    workclass     : str
    education     : str
    marital_status: str = Field(alias='marital-status')
    occupation    : str
    relationship  : str
    race          : str
    sex           : str
    native_country: str = Field(alias='native-country')


@app.get("/")
def read_root():
    return {"Welcome": "Home"}


@app.post("/predict")
def predict_salary_level(item: DataItem):
    item_dict = item.dict()
    X = pd.DataFrame([[item_dict[col] for col in cols]], columns=cols)

    X_categorical = X[cat_features].values
    X_continuous = X.drop(*[cat_features], axis=1)

    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)


    pred = int(model.predict(X)[0])

    return {"prediction": pred}