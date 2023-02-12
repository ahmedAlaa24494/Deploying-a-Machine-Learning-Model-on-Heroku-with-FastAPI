"""Test preprocessing and inference steps in the pipeline
"""
import pandas as pd
import pytest
import yaml
import joblib
import sys
from pathlib import Path

src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.ml_model.data import process_data
from src.ml_model.model import inference


@pytest.fixture
def clean_data():
    """get clean data"""
    data = pd.read_csv("data/clean_census.csv")
    return data


@pytest.fixture
def config():
    """Read Configuraion File"""
    with open("params.yaml") as conf_obj:
        config = yaml.safe_load(conf_obj)
    return config


def test_x_y_equalized(config):
    train_df = pd.read_csv(config["data_split"]["output"] + "/train_set.csv")
    x, y, label_encoder, cat_features_encoder, num_features_encoder = process_data(
        X=train_df,
        target=config["preprocess"]["target"],
        do_train=True,
        cat_features=config["preprocess"]["categorical_features"],
        num_features=config["preprocess"]["numerical_features"],
    )
    assert x.shape[0] == y.shape[0]


def test_fixed_encoders(clean_data, config):
    base_label_encoder = joblib.load(
        f"{config['preprocess']['encoders_dir']}/label_encoder.gz"
    )
    base_cat_encoder = joblib.load(
        f"{config['preprocess']['encoders_dir']}/cat_features_encoder.gz"
    )
    base_num_encoder = joblib.load(
        f"{config['preprocess']['encoders_dir']}/num_features_encoder.gz"
    )

    _, _, label_encoder, cat_features_encoder, num_features_encoder = process_data(
        X=clean_data,
        target=config["preprocess"]["target"],
        do_train=True,
        cat_features=config["preprocess"]["categorical_features"],
        num_features=config["preprocess"]["numerical_features"],
    )
    _, _, _, _, _ = process_data(
        X=clean_data,
        target=config["preprocess"]["target"],
        do_train=False,
        cat_features=config["preprocess"]["categorical_features"],
        num_features=config["preprocess"]["numerical_features"],
        label_encoder=base_label_encoder,
        cat_features_encoder=base_cat_encoder,
        num_features_encoder=base_num_encoder,
    )

    assert label_encoder.get_params() == base_label_encoder.get_params()
    assert cat_features_encoder.get_params() == base_cat_encoder.get_params()
    assert num_features_encoder.get_params() == base_num_encoder.get_params()


def test_inference(config):
    label_encoder = joblib.load(
        f"{config['preprocess']['encoders_dir']}/label_encoder.gz"
    )
    cat_features_encoder = joblib.load(
        f"{config['preprocess']['encoders_dir']}/cat_features_encoder.gz"
    )
    num_features_encoder = joblib.load(
        f"{config['preprocess']['encoders_dir']}/num_features_encoder.gz"
    )
    gb_model = joblib.load(config["train"]["model_artifact"])

    example = [
        {
            "age": 28,
            "workclass": "Private",
            "education": "HS-grad",
            "marital-status": "Separated",
            "occupation": "Sales",
            "relationship": "Other-relative",
            "race": "White",
            "sex": "Female",
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    ]

    x_ex, _, _, _, _ = process_data(
        X=pd.DataFrame(example),
        target=None,
        do_train=False,
        cat_features=config["preprocess"]["categorical_features"],
        num_features=config["preprocess"]["numerical_features"],
        label_encoder=label_encoder,
        cat_features_encoder=cat_features_encoder,
        num_features_encoder=num_features_encoder,
    )

    prediction = label_encoder.inverse_transform(inference(gb_model, x_ex))[0]

    assert isinstance(prediction, str)
    assert prediction == "<=50K"
