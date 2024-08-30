'''
this file is used to test the model creation'''
import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from model_exploration.model_creation import load_data, clean_data, split_data, main


def test_load_data():    
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0


