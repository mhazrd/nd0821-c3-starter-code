from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Welcome": "Home"}


def test_predict_positive():
    item = {'age': 42, 'workclass': 'Private', 'fnlgt': 159449, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 5178, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'}
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": 1}


def test_predict_negative():
    item = {'age': 39, 'workclass': 'State-gov', 'fnlgt': 77516, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Never-married', 'occupation': 'Adm-clerical', 'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Male', 'capital-gain': 2174, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'}
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": 0}
