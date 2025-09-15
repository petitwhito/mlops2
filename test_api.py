import requests

# Test endpoint predict
data = {
    "Age": 50,
    "Sex": 1,
    "ChestPainType": 1, 
    "RestingBP": 140,
    "Cholesterol": 250,
    "FastingBS": 0,
    "RestingECG": 1,
    "MaxHR": 150,
    "ExerciseAngina": 0,
    "Oldpeak": 1.0,
    "ST_Slope": 1
}

response = requests.post("http://127.0.0.1:8000/predict", params=data, timeout=30)
print("Status code:", response.status_code)
print("Response text:", response.text)
if response.status_code == 200:
    print("Predict response:", response.json())

response = requests.post("http://127.0.0.1:8000/update-model", timeout=30)
print("Update model response:", response.json())
