import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    """App client to be tested"""
    cl = TestClient(app)
    return cl


def test_get_request(client):
    req = client.get("/")
    assert req.status_code == 200
    assert req.json() == {"message": "Greetings!"}


def test_post_request(client):
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
    req = client.post("/", json=person)
    assert req.status_code == 200
    assert req.json() == {"Prediction": "<=50K"}


def test_wrong_requests(client):
    wrong_info = {
        "age": 20,
        "workclass": "Private",
        "education": "Some-college",
        "marital_status": "Extremly Single",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "hours_per_week": 60,
        "native_country": "United-States",
    }
    req = client.post("/", json=wrong_info)
    assert req.status_code != 200
