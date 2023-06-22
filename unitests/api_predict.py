import json
import requests
import pandas as pd

SAMPLE_DATA_PATH = './../st_content/sample_data.csv'

API_URI = "https://oc-ds-p7.herokuapp.com/solvability_prediction"


def api_call():
    status = 0
    df = pd.read_csv(SAMPLE_DATA_PATH)
    X = df.drop(columns=['SK_ID_CURR'])
    associated_data = X.iloc[0, :].values.tolist()
    dict_customer = {k: v for k, v in zip(df.columns, associated_data)}
    input_json = json.dumps(dict_customer)
    response = requests.post(API_URI, data=input_json)
    if (response.text == '0') | (response.text == '1'):
        status = 1
    return status


def test_answer():
    assert 1 == api_call()
