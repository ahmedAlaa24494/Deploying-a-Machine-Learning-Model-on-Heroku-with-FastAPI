""" Testing basic cleaning module
"""
import pandas as pd
import pytest

@pytest.fixture
def clean_data():
    """get clean data
    """
    data = pd.read_csv("data/clean_census.csv")
    return data

def test_unwanted_values(clean_data):
    """Check if `?` exist in the dataset
    """
    assert '?' not in clean_data.values

def test_null_values(clean_data): 
    """Testing if any row had None value
    """
    assert clean_data.dropna().shape == clean_data.shape

