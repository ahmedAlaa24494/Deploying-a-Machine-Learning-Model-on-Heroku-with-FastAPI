""" Preprocess Census data 
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler


def process_data(
    X: pd.DataFrame,
    target: str = None,
    cat_features: list = [],
    num_features: list = [],
    do_train: bool = True,
    label_encoder: LabelBinarizer = None,
    cat_features_encoder: OneHotEncoder = None,
    num_features_encoder: StandardScaler = None,
):
    """Preprocess cleaned Census dataframes, for train or inference

    Args:
        X (pd.DataFrame):
                DataFrame to be processed
        target (str, optional):
                The target column to be predicted (Classification Target). Defaults to None.
        do_train (bool):
                if True the target values must be provided for training or testing,
                else Label Encode and cat_features_encoder must be given. Defaults to True
        cat_features (list, optional):
                Categorical features names in the DataFrame.
                If list empty use all the features on the Frame Defaults to [].
        cat_features (list, optional):
                Numerical features names in the DataFrame.
                If list empty use all the features on the Frame Defaults to [].
        label_encoder (LabelBinarizer, optional):
                The Label Encoder,
                For training case this Encoder must be initialized and fitted on the train set,
                For Inference LabelEncoder must be per-fitted  . Defaults to None.
        cat_features_encoder (OneHotEncoder, optional): Categorical features encoder,
                For training case this Encoder must be initialized and fitted on the train set,
                For Inference LabelEncoder must be per-fitted  . Defaults to None.
        num_features_encoder (OneHotEncoder, optional): Numerical features encoder,
                For training case this Encoder must be initialized and fitted on the train set,
                For Inference LabelEncoder must be per-fitted  . Defaults to None.
    """
    if target is not None:
        y = X[target]
        x = X.drop([target], axis=1)
    else:
        y = np.array([])

    x_cat = X[cat_features].values
    x_num = X[num_features].values

    if do_train:
        cat_features_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        num_features_encoder = StandardScaler()
        label_encoder = LabelBinarizer()
        x_cat = cat_features_encoder.fit_transform(x_cat)
        x_num = num_features_encoder.fit_transform(x_num)
        y = label_encoder.fit_transform(y.values).ravel()
    else:
        x_cat = cat_features_encoder.transform(x_cat)
        x_num = num_features_encoder.transform(x_num)

        if y.shape[0] == x_cat.shape[0]:
            y = label_encoder.transform(y)

    ## Concat categorical and numerical features
    x = np.concatenate([x_cat, x_num], axis=1)
    return x, y, label_encoder, cat_features_encoder, num_features_encoder
