import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("sarimax_model.pkl")
df = joblib.load("df.pkl")

class InputData(BaseModel):
    number_of_dates: int

app = FastAPI()

def get_forecast_values(results, df, N):
    last_date = df.reset_index().at[len(df)-1, 'date']  # extracting the last date
    print(f"The last date is: {last_date}")
    forecast_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
    print(f"start forecast date: {forecast_date}")
    forecast = results.forecast(steps=N)  # forecasting for N days
    forecast_index = pd.date_range(start=forecast_date, periods=N)
    df_forecast = pd.DataFrame({'forecast': forecast.values}, index=forecast_index)
    return df_forecast

@app.post("/predict")
async def predict(data: InputData):
    df_forecast = get_forecast_values(model, df, N=data.number_of_dates)

    return df_forecast.to_dict()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)