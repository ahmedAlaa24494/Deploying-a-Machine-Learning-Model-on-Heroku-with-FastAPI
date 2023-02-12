import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from pathlib import Path
import joblib
import sys
import pandas as pd

src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.utils.logs import get_logger
from src.ml_model.data import process_data
from src.ml_model.model import compute_model_metrics


def slice_validation(config_path: Text) -> None:
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
    test_df = pd.read_csv("data/train/test_set.csv")
    cat_features = config["preprocess"]["categorical_features"]
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
    logger.info("Compute slice metrices")
    slice_metricies = []
    for cat in cat_features:
        ## Loop over unique values of each categorical feature
        for slice in test_df[cat].unique():
            df_slice = test_df[test_df[cat] == slice]
            (
                x,
                y,
                label_encoder,
                cat_features_encoder,
                num_features_encoder,
            ) = process_data(
                X=df_slice,
                target=config["preprocess"]["target"],
                do_train=False,
                cat_features=config["preprocess"]["categorical_features"],
                num_features=config["preprocess"]["numerical_features"],
                label_encoder=label_encoder,
                cat_features_encoder=cat_features_encoder,
                num_features_encoder=num_features_encoder,
            )
            y_preds = gb_model.predict(x)
            precision, recall, fbeta = compute_model_metrics(preds=y_preds, y=y)
            line = "[%s->%s] Precision: %s " "Recall: %s FBeta: %s" % (
                cat,
                slice,
                precision,
                recall,
                fbeta,
            )
            logger.info(line)
            slice_metricies.append(line)

    logger.info("Slice metricies computed")
    with open("reports/slice_output.txt", "w") as out:
        for slice_value in slice_metricies:
            out.write(slice_value + "\n")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    slice_validation(config_path=args.config)
