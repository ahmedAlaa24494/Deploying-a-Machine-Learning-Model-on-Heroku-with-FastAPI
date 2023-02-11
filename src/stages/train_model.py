import argparse
import joblib
from typing import Text
from collections import ChainMap
import yaml
import json
from pathlib import Path
import sys 
import numpy as np
src_dir = Path.cwd()
sys.path.append(str(src_dir)) 
from src.utils.logs import get_logger
from src.ml_model.model import train_model, compute_model_metrics

def train(config_path: Text) -> None: 
    """
    Load and validate data
    """ 

    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('Train Model', log_level=config['base']['log_level'])
    logger.info("Read train and test data")
    x_train , y_train = np.load(config['preprocess']['output_dir']+'/x_train.npy'), \
                    np.load(config['preprocess']['output_dir']+'/y_train.npy') 

    x_test , y_test = np.load(config['preprocess']['output_dir']+'/x_test.npy'), \
                        np.load(config['preprocess']['output_dir']+'/y_test.npy') 

        
    # Preprocess
    logger.info("Preprocess datasets")
    ## Train GradientBoostingClassifier
    logger.info("Training GradientBoostingClassifier")
    params = dict(ChainMap(*config['train']['hp']))
    gb_model = train_model(x_train, y_train, params)
    ## Generate test estimations
    logger.info("Compute test estimations with the trained model")
    preds = gb_model.predict(x_test)
    ## Compute Evaluation metrices 
    precision, recall, fbeta = compute_model_metrics(preds=preds, y=y_test)
    metrics = {"Model Name":"GradientBoostingClassifier" ,"precision":precision, "recall":recall, "fbeta":fbeta}
    logger.info("Test Metrics\n"+str(metrics))
    with open("reports/metrics.json",'w')as obj:
        json.dump(metrics,obj)

    joblib.dump(gb_model, config['train']['model_artifact'])


    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train(args.config)