from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
from keras.models import load_model
import zipfile
import numpy as np
from scipy.io import wavfile

app = FastAPI()

generator = load_model("audio_generator.h5")

SAMPLE_RATE = 16000

class AudioFile(BaseModel):
    file: bytes
    filename: str

@app.get("/generate_audio")
async def generate_audio(num_files: int = Query(1, ge=1)):
    audio_samples = []
    for _ in range(num_files):
        noise = np.random.normal(0, 1, (1, 100))
        audio = generator.predict(noise)[0]
        audio_samples.append(audio)

    # Create an in-memory zip archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, audio in enumerate(audio_samples):
            file_name = f"generated_audio_{i}.wav"
            wavfile.write(file_name, SAMPLE_RATE, audio)
            zip_file.write(file_name)

    # Set the buffer position to the beginning
    zip_buffer.seek(0)

    # Return the zip archive as a streaming response
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment;filename=generated_audio.zip"})