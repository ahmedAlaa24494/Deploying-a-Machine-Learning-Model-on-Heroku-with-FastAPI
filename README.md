# ML Pipeline with CI/CD usign github and Heroku
Udacity project about creating a pipeline to train a model and deplot it with a public API on Heroku.

## Model testing
The testing is automated with each dvc exp run to validate expremint stages 
```
pytest src/stages/test_data_cleaning.py  src/stages/test_model_steps.py 
```
## API Testing
Test API Get and Post requests
```
pytest test_api.py
python test_heroku_api.py
```

## Run and test new experiment with DVC
```
dvc repro
```
OR
```
dvc exp run -f
```

## Serve the API
- Test API locally with ```uvicorn api_server:app --reload```
- Test heroku API with ```python test_heroku_api.py```

## Screenshots requested:
* [dvc-dag.png](https://github.com/ahmedAlaa24494/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/blob/master/screenshots/dvc-dag.png)
* [continuous_deloyment.png](https://github.com/ahmedAlaa24494/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/blob/master/screenshots/continuous_deloyment.png)
* [continuous_integration.png](https://github.com/ahmedAlaa24494/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/blob/master/screenshots/continuous_integration.png)
* [live_get.png](https://github.com/ahmedAlaa24494/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/blob/master/screenshots/live_get.png)
* [live_post.png](https://github.com/ahmedAlaa24494/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/blob/master/screenshots/live_post.png)
