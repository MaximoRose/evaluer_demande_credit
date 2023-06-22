import json
import requests
import pandas as pd

SAMPLE_DATA_PATH = './st_content/bad_sample_data.csv'

API_URI = "https://oc-ds-p7.herokuapp.com/preprocess_data"


def api_call():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    X = df.drop(columns=['SK_ID_CURR'])
    associated_data = X.iloc[0, :].values.tolist()
    dict_customer = {k: v for k, v in zip(df.columns, associated_data)}
    input_json = json.dumps(dict_customer)
    response = requests.post(API_URI, data=input_json)
    try:
        # Cette cle de premier nineau n'est presente que dans des messages d'erreur
        json.loads(response.text)['detail']
        return 0
    except KeyError:
        return 1


def test_answer():
    assert 1 == api_call()
