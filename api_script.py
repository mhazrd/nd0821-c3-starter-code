import requests


host = 'https://udacity-mlop-test.herokuapp.com'
predict_uri = f"{host}/predict"

item = {'age': 42, 'workclass': 'Private', 'fnlgt': 159449, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 5178, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'}
    
r = requests.post(predict_uri, json=item)

print('Request body: ', item)
print('Status code: ', r.status_code)
print('Reponse: ', r.json())