import os
import io
import numpy as np
import sounddevice as sd
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import librosa

# Constants
SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]

app = FastAPI(title="Urban Sound Classifier API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('models/urban_sound_cnn_crnn.h5')

def preprocess_audio(audio_data, sr):
    """Preprocess audio data for model input."""
    # Resample if necessary
    if sr != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Pad or trim to fixed length
    if len(audio_data) > SAMPLE_RATE * DURATION:
        audio_data = audio_data[:SAMPLE_RATE * DURATION]
    else:
        audio_data = np.pad(audio_data, (0, max(0, SAMPLE_RATE * DURATION - len(audio_data))))
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add batch dimension
    return np.expand_dims(log_mel_spec, axis=0)

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "OK"}

@app.get("/classes")
async def get_classes():
    """Return list of available sound classes."""
    return {"classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict class from uploaded audio file."""
    # Read and preprocess audio file
    audio_data, sr = librosa.load(io.BytesIO(await file.read()), sr=None)
    features = preprocess_audio(audio_data, sr)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = CLASS_NAMES[prediction.argmax()]
    
    return {"predicted_class": predicted_class}

@app.post("/record")
async def record():
    """Record audio and predict class."""
    try:
        # Record audio
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1
        )
        sd.wait()
        
        # Preprocess audio
        features = preprocess_audio(recording.flatten(), SAMPLE_RATE)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = CLASS_NAMES[prediction.argmax()]
        
        return {"predicted_class": predicted_class}
        
    except Exception as e:
        return {"error": str(e)}