import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
import joblib
import sys
import pandas as pd
import numpy as np

src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.utils.logs import get_logger
from src.ml_model.data import process_data


def preprocess(config_path: Text) -> None:
    """
    Load and validate data
    """

    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("Preprocess Data", log_level=config["base"]["log_level"])
    logger.info("Read train and test files")
    train_df = pd.read_csv("data/train/train_set.csv")
    test_df = pd.read_csv("data/train/test_set.csv")
    # Preprocess training data and create features and target encoders
    logger.info("Preprocess train dataset")
    (
        x_train,
        y_train,
        label_encoder,
        cat_features_encoder,
        num_features_encoder,
    ) = process_data(
        X=train_df,
        target=config["preprocess"]["target"],
        do_train=True,
        cat_features=config["preprocess"]["categorical_features"],
        num_features=config["preprocess"]["numerical_features"],
    )
    logger.info("Preprocess test dataset")
    (
        x_test,
        y_test,
        label_encoder,
        cat_features_encoder,
        num_features_encoder,
    ) = process_data(
        X=test_df,
        target=config["preprocess"]["target"],
        do_train=False,
        cat_features=config["preprocess"]["categorical_features"],
        num_features=config["preprocess"]["numerical_features"],
        label_encoder=label_encoder,
        cat_features_encoder=cat_features_encoder,
        num_features_encoder=num_features_encoder,
    )

    ## Save encoders
    logger.info("Saving Encoders")
    joblib.dump(
        label_encoder, f"{config['preprocess']['encoders_dir']}/label_encoder.gz"
    )
    joblib.dump(
        cat_features_encoder,
        f"{config['preprocess']['encoders_dir']}/cat_features_encoder.gz",
    )
    joblib.dump(
        num_features_encoder,
        f"{config['preprocess']['encoders_dir']}/num_features_encoder.gz",
    )
    ## Save Features
    logger.info("Saving Features")
    np.save(f"{config['preprocess']['output_dir']}/x_train.npy", x_train)
    np.save(f"{config['preprocess']['output_dir']}/y_train.npy", y_train)
    np.save(f"{config['preprocess']['output_dir']}/x_test.npy", x_test)
    np.save(f"{config['preprocess']['output_dir']}/y_test.npy", y_test)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    preprocess(config_path=args.config)
