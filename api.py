import os
from typing  import Literal
from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import pandas as pd
from typing import Literal
import joblib
from pandas.core.frame import DataFrame
from src.ml_model.model import inference
from src.ml_model.data import process_data
# Load config file with yaml
with open("params.yaml") as f:
    config= yaml.safe_load(f)



class Person(BaseModel): 
    age: int

    workclass: Literal['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Local-gov',
        'Federal-gov', 'State-gov', 'Without-pay']

    education: Literal['Some-college', 'HS-grad', '5th-6th', 'Bachelors', '7th-8th',
        'Doctorate', '11th', 'Masters', 'Assoc-voc', 'Prof-school', '10th',
        '1st-4th', 'Assoc-acdm', '9th', 'Preschool', '12th']

    marital_status: Literal['Divorced', 'Never-married', 'Married-civ-spouse', 'Widowed',
        'Separated', 'Married-AF-spouse', 'Married-spouse-absent']

    occupation: Literal['Exec-managerial', 'Other-service', 'Sales', 'Transport-moving',
        'Craft-repair', 'Machine-op-inspct', 'Prof-specialty',
        'Adm-clerical', 'Handlers-cleaners', 'Tech-support',
        'Protective-serv', 'Farming-fishing', 'Priv-house-serv',
        'Armed-Forces']

    relationship: Literal['Unmarried', 'Not-in-family', 'Own-child', 'Husband', 'Wife',
        'Other-relative']

    race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Other',
        'Amer-Indian-Eskimo']

    sex: Literal['Female', 'Male']

    hours_per_week: int
    native_country: Literal['United-States', 'Thailand', 'Guatemala', 'Puerto-Rico', 'Canada',
        'India', 'Vietnam', 'Mexico', 'Italy', 'Cambodia', 'Haiti',
        'El-Salvador', 'Germany', 'Peru', 'China', 'Honduras',
        'Philippines', 'England', 'Ecuador', 'Hong', 'Portugal', 'France',
        'Ireland', 'Laos', 'South', 'Jamaica', 'Greece', 'Cuba',
        'Nicaragua', 'Japan', 'Columbia', 'Poland', 'Dominican-Republic',
        'Trinadad&Tobago', 'Hungary', 'Scotland', 'Taiwan',
        'Holand-Netherlands', 'Iran', 'Outlying-US(Guam-USVI-etc)',
        'Yugoslavia']


app = FastAPI()
@app.get("/")
async def get_items():
    return {"message": "Greetings!"}


@app.post("/")
async def infer(person:Person):
    from src.ml_model.model import inference
    label_encoder = joblib.load(f"{config['preprocess']['encoders_dir']}/label_encoder.gz")
    cat_features_encoder = joblib.load(f"{config['preprocess']['encoders_dir']}/cat_features_encoder.gz")
    num_features_encoder = joblib.load(f"{config['preprocess']['encoders_dir']}/num_features_encoder.gz")
    model = joblib.load(config['train']['model_artifact'])

    person_dict = [{'age': person.age,
    'workclass': person.workclass,
    'education': person.education,
    'marital-status': person.marital_status,
    'occupation': person.occupation,
    'relationship': person.relationship,
    'race': person.race,
    'sex': person.sex,
    'hours-per-week': person.hours_per_week,
    'native-country': person.native_country,
    }]

    x, _, _, _, _ = process_data(X=pd.DataFrame(person_dict),
        target=None,
        do_train=False,
        cat_features=config['preprocess']['categorical_features'], 
        num_features=config['preprocess']['numerical_features'], 
        label_encoder=label_encoder,
        cat_features_encoder=cat_features_encoder,
        num_features_encoder=num_features_encoder)
    prediction = label_encoder.inverse_transform(inference(model, x))[0] 
    return {"Prediction":prediction}