import argparse
import pandas as pd
from typing import Text
import yaml
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

src_dir = Path.cwd()
sys.path.append(str(src_dir))
from src.utils.logs import get_logger


def data_split(config_path: Text) -> None:
    """
    Load and validate data
    """

    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger("Split dataset", log_level=config["base"]["log_level"])
    data = pd.read_csv(config["data_split"]["dataset"])

    train_d, test_d = train_test_split(
        data, test_size=config["data_split"]["test_size"]
    )
    train_d.to_csv("data/train/train_set.csv", index=False)
    test_d.to_csv("data/train/test_set.csv", index=False)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    data_split(config_path=args.config)
