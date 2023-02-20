"""
Post Heroku API
"""
import requests


person = {
    "age": 20,
    "workclass": "Private",
    "education": "Some-college",
    "marital_status": "Divorced",
    "occupation": "Exec-managerial",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Female",
    "hours_per_week": 60,
    "native_country": "United-States",
}
r = requests.post('https://census-estimator.herokuapp.com/', json=person)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())