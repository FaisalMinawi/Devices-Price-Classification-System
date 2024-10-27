from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the model
with open('device_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.post("/predict_price/")
async def predict_price(data: dict):
    input_data = pd.DataFrame([data])
    print(input_data)
    prediction = model.predict(input_data)
    print(prediction)
    return {"predicted_price_range": int(prediction[0])}
