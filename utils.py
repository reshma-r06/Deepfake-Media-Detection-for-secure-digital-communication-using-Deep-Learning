import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess

def extract_features(y, sr, max_features=34):
    """Extract exactly 34 audio features consistently"""
    # Normalize audio
    y = librosa.util.normalize(y)
    features = []
    # 1. Basic audio properties (4 features)
    features.append(np.mean(y))
    features.append(np.std(y))
    features.append(librosa.feature.zero_crossing_rate(y=y)[0, 0])
    features.append(librosa.feature.rms(y=y)[0, 0])
    # 2. Spectral features (30 features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Take first 10 MFCCs (10 features)
    features.extend(mfccs[:10, 0].tolist())
    # Take first 12 chroma features (12 features)
    features.extend(chroma[:12, 0].tolist())
    # Take first 8 spectral contrast (8 features)
    features.extend(spectral_contrast[:8, 0].tolist())
    # Ensure exactly max_features
    if len(features) < max_features:
        features.extend([0.0] * (max_features - len(features)))
    return np.array(features[:max_features])

def process_dataset(data_dir='data', output_csv='audio_features.csv'):
    """Process all videos and save features"""
    temp_dir = os.path.join(os.getcwd(), 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)
    features = []
    labels = []
    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(data_dir, folder)
        print(f"Processing {folder} videos...")
        for video_file in tqdm(os.listdir(folder_path)):
            if not video_file.endswith('.mp4'):
                continue
            video_path = os.path.join(folder_path, video_file)
            temp_path = os.path.join(temp_dir, f"temp_{video_file}.wav")
            try:
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '44100', '-ac', '1',
                    '-y', temp_path
                ], check=True, stderr=subprocess.PIPE)
                y, sr = librosa.load(temp_path, sr=None)
                os.unlink(temp_path)
                # Extract features
                feat = extract_features(y, sr)
                features.append(feat)
                labels.append(label)
            except subprocess.CalledProcessError as e:
                print(f"\nFailed to process {video_file}: {e.stderr.decode()}")
                continue
            except Exception as e:
                print(f"\nError processing {video_file}: {str(e)}")
                continue
    # Clean up temp directory
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    # Save to CSV
    if features:
        df = pd.DataFrame(features)
        df['label'] = labels
        df.to_csv(output_csv, index=False)
        print(f"\nSuccess! Saved {len(features)} samples to {output_csv}")
    else:
        print("\nError: No features were extracted!")

if __name__ == '__main__':
    process_dataset()