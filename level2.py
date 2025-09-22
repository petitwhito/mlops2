"""
We now want to have a web service service to serve our machine learning model.

The web service should load the model model from the MLFlow server upon server start.

The webservice should have the following endpoints : 

    - a /predict endpoint to return predictions upon a Post request
    - a /update-model endpoint  allowing to  update the model with a webflow model version

     
    1. Build such service with fastAPI or flask, or any other service you want. 
    2. Make a script to test automatically test your /predic endpoint
    3. Test your update-model endpoint to be sure that the model is updated after calling the endpoint.
    4. Package your webservice in  a docker container. You should use docker compose to launch your container


Remark : the model should not be saved in the Dockerfile since it is loaded by the service from mlflow 


Part 3 - Canary deployment

We would like now to do canary deployment. 

For that we will have two loaded models : 

    current (for the current model version)
    next (for the next version we would like to have)

At startup both current and next model should be the same. 

the /predict version should use the current model with a p probability and the next model with 1 - p. 

the /update-model endpoint should update the next model 

the /accept-next-model endpoint should set next-model as current so that both current and next are used

"""

import fastapi as fa
import mlflow
import os
import random
import pandas as pd

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
mlflow.set_tracking_uri(uri=mlflow_uri)

def load_model_from_mlflow(version="1"):
    try:
        return mlflow.sklearn.load_model(f"models:/heart_disease_model/{version}")
    except:
        return mlflow.sklearn.load_model("models:/heart_disease_model/latest")

current_model = None
next_model = None
p = 0.8  

app = fa.FastAPI()

@app.post("/predict")
def predict(Age: int, Sex: int, ChestPainType: int, RestingBP: int, Cholesterol: int, FastingBS: int, RestingECG: int, MaxHR: int, ExerciseAngina: int, Oldpeak: float, ST_Slope: int):
    global current_model, next_model
    if current_model is None:
        current_model = load_model_from_mlflow()
        next_model = load_model_from_mlflow()
    
    if random.random() < p:
        model = current_model
        model_used = "current"
    else:
        model = next_model
        model_used = "next"
    
    df = pd.DataFrame([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]], 
                      columns=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0]), "model_used": model_used}
  
@app.post("/update-model")
def update_model():
    global next_model
    next_model = load_model_from_mlflow("latest")
    return {"message": "Next model updated"}

@app.post("/accept-next-model")
def accept_next_model():
    global current_model, next_model
    current_model = next_model
    return {"message": "Next model accepted as current"}