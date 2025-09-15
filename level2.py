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
"""

import fastapi as fa
import mlflow
import os

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
mlflow.set_tracking_uri(uri=mlflow_uri)

def load_model_from_mlflow(version="1"):
    try:
        return mlflow.sklearn.load_model(f"models:/heart_disease_model/{version}")
    except:
        return mlflow.sklearn.load_model("models:/heart_disease_model/latest")

model = None
app = fa.FastAPI()

@app.post("/predict")
def predict(Age: int, Sex: int, ChestPainType: int, RestingBP: int, Cholesterol: int, FastingBS: int, RestingECG: int, MaxHR: int, ExerciseAngina: int, Oldpeak: float, ST_Slope: int):
    global model
    if model is None:
        model = load_model_from_mlflow()
    prediction = model.predict([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
    return {"prediction": int(prediction[0])}
  
@app.post("/update-model")
def update_model():
    global model
    model = load_model_from_mlflow("latest")
    return {"message": "Model updated"}