"""
This module will conduct test for testing smooth operation of Flask Application
"""

# import flask app object from python file containing it
from main import app
import pytest

@pytest.fixture # <-- Hardcoded, do not change
def client():
    return app.test_client() # <-- Hardcoded, do not change

def test_home(client):
    response=client.get('/')
    assert response.status_code == 200

def test_pearson(client):
    response=client.get('/pearson')
    assert response.status_code == 200
    response = client.post('/pearson', json={'user':1,'n_recommend':10})
    assert response.status_code == 200

def test_cosine(client):
    response=client.get('/cosine')
    assert response.status_code == 200
    response = client.post('/cosine', json={'user':1,'n_recommend':10})
    assert response.status_code == 200