import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    avg_credit_limit: float
    total_credit_cards: float

app = FastAPI()

@app.post("/predict")
async def predict(data: InputData):
    # Transform the input data
    x = np.array([data.avg_credit_limit, data.total_credit_cards]).reshape(1, -1)
    x = scaler.transform(x)

    pred = model.predict(x)[0]

    return {"cluster_label": int(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)