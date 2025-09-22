"""
  1.1 Using the MLflow documentation (here for instance), make a model training script on a simple dataset of your choice. 

The model training script should track : 

    - the the model hyper-parameters
    - the model metric (mse, accuracy or whatever)

No need to version the model for now

 

1.2 Run the script and check that the tracked data are available in mlflow UI

 

1.3 Change one hyper parameter of the model and rerun the training script. Check that both this run and the previous one are availables.

 

1.4 Make sure that the model is also saved in MLFlow. Rerun the training script and access it in the interface
"""

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Sklearn version: ", sklearn.__version__)

import torch
from torch import cuda
print("Torch version: ", torch.__version__)
print("Cuda version: ", cuda.get_device_name(0))

import mlflow
from mlflow.models import infer_signature

print("Mlflow version: ", mlflow.__version__)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow-Heart-Disease-Level-1")

print("Experiment: ", mlflow.get_experiment_by_name("MLflow-Heart-Disease-Level-1"))

data = pd.read_csv("heart.xls")
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data[["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]]
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"solver": "lbfgs", "max_iter": 5000, "multi_class": "auto", "random_state": 8888}
model = LogisticRegression(**params)

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    mlflow.log_params(params)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "heart_disease_model", signature=infer_signature(X_train, y_train))
    
    model_uri = f"runs:/{run.info.run_id}/heart_disease_model"
    mlflow.register_model(model_uri, "heart_disease_model")

print(f"Accuracy: {accuracy}")