import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Constants
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048

def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load and preprocess audio file."""
    try:
        # Load audio file
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad or trim to fixed length
        if len(y) > sr * duration:
            y = y[:sr * duration]
        else:
            y = np.pad(y, (0, max(0, sr * duration - len(y))))
        
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def extract_features(y):
    """Convert audio to log-mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def main():
    """Main preprocessing function."""
    # Set paths
    data_dir = "../data/UrbanSound8K"
    metadata_path = os.path.join(data_dir, "metadata/UrbanSound8K.csv")
    audio_dir = os.path.join(data_dir, "audio")
    
    # Load metadata
    print("Loading metadata...")
    metadata = pd.read_csv(metadata_path)
    
    # Initialize arrays for features and labels
    X = []
    y = []
    
    print("Processing audio files...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = os.path.join(audio_dir, f'fold{row["fold"]}', row["slice_file_name"])
        
        # Load and preprocess audio
        audio = load_audio(file_path)
        if audio is None:
            continue
        
        # Extract features
        features = extract_features(audio)
        
        X.append(features)
        y.append(row["classID"])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Save processed data
    print("Saving processed data...")
    np.save("../data/X.npy", X)
    np.save("../data/y.npy", y)
    
    print(f"Preprocessing complete. Saved {len(X)} samples.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    main()