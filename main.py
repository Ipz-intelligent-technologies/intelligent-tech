import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

from fastapi import FastAPI
from fastapi import File, UploadFile

from io import BytesIO

import numpy as np

app = FastAPI()

MODEL = tf.keras.models.load_model('model.kf')

@app.get("/ping")
def root():
    return "pong"

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    file = await file.read()
    img = load_img(BytesIO(file), target_size=(150, 150))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x]) 

    classes = MODEL.predict(images, batch_size=10)

    if classes[0]>0:
        return {"prediction": "This is a dog"}
    else:
        return {"prediction": "This is a cat"}