import pandas as pd


def clean_cencus(data:pd.DataFrame)->pd.DataFrame:
    """
    Achieve the basic cleaning for cencus data frame
    """ 
    data.replace({'?':None}, inplace=True)
    data.dropna(inplace=True)
    data.drop("fnlgt", axis="columns", inplace=True)
    data.drop("education-num", axis="columns", inplace=True)
    data.drop("capital-gain", axis="columns", inplace=True)
    data.drop("capital-loss", axis="columns", inplace=True)
    return data.reset_index(drop=True)