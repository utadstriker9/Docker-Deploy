import io
import pandas as pd
import requests


@data_loader
def load_data_from_api(*args, **kwargs):
    path = 'magic/data_loaders/dataset_user_behavior_for_test.csv'
    return pd.read_csv(path, sep=',')


@test
def test_row_count(df, *args) -> None:
    assert len(df.index) >= 1000, 'The data does not have enough rows.'
