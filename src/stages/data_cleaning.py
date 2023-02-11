import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from colorama import Fore
from tqdm import tqdm
from pathlib import Path
import sys 
src_dir = Path.cwd()
sys.path.append(str(src_dir)) 
from src.utils.logs import get_logger
from src.utils.data_clean import clean_cencus


def load_clean_data(config_path: Text) -> None: 
    """
    Load and validate data
    """ 

    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('Data Cleaning', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    ## Read relations file
    data = pd.read_csv(config['load_clean']['dataset'])
    logger.info("Cleaning data!!!")
    data = clean_cencus(data)
    logger.info("Saving cleaned data")
    data.to_csv(config['load_clean']['output'],index=False)

    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    load_clean_data(config_path=args.config)