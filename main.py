from fastapi import FastAPI, UploadFile
import tensorflow as tf
import librosa
import tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler


app = FastAPI()

# Step 1: Load the trained model
model = tf.keras.models.load_model("res_model.h5")

def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_features = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_features)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                       ))
    return result


# Step 2: Preprocess the received audio file
def preprocess_audio(audio_data):
    received_audio, received_sr = librosa.load(audio_data, duration=2.5, offset=0.6)
    preprocessed_audio = extract_features(received_audio, received_sr)
    return preprocessed_audio

# Step 3: Normalize the extracted features
def normalize_features(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform([features])
    return normalized_features

# Step 4: Classify the emotion
def classify_emotion(features):
    normalized_features = normalize_features(features)
    predicted_probabilities = model.predict(normalized_features)
    return predicted_probabilities

# Step 5: Determine the predicted emotion
def get_predicted_emotion(probabilities):
    emotion_labels = ['disgust', 'fear', 'sad', 'neutral', 'happy', 'angry', 'surprise']
    predicted_emotion_index = np.argmax(probabilities)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    return predicted_emotion

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(await file.read())

    preprocessed_audio = preprocess_audio(contents)
    predicted_probabilities = classify_emotion(preprocessed_audio)
    predicted_emotion = get_predicted_emotion(predicted_probabilities)
    return {"predicted_emotion": predicted_emotion}