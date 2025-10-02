import os
import tempfile
from flask import Flask, render_template, request
import librosa
from utils import extract_features
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import subprocess

app = Flask(__name__, template_folder='templates')

# Load model and scaler
model = load_model('models/audio_model.h5')
scaler = joblib.load('models/scaler.save')

# Create required directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('temp_audio', exist_ok=True)

## Removed local extract_features; now using the one from utils.py

def predict_audio(filepath):
    import shlex
    try:
        # Use pathlib for safer path handling
        from pathlib import Path
        input_path = Path(filepath)
        temp_wav = Path('temp_audio') / (input_path.stem + '.wav')

        # Ensure paths are strings and quoted for ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            '-y',
            str(temp_wav)
        ]
        print("Running FFmpeg command:", ' '.join(shlex.quote(str(x)) for x in ffmpeg_cmd))
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg stderr:", result.stderr)
            return f"FFmpeg Error: {result.stderr}", 0.0

        # Load audio and extract features
        y, sr = librosa.load(str(temp_wav), sr=None)
        features = extract_features(y, sr)

        # Clean up
        temp_wav.unlink(missing_ok=True)

        # Scale features and predict
        features_scaled = scaler.transform([features])
        prob = model.predict(features_scaled)[0][0]
        print(f"Predicted probability: {prob}")
        # Swap logic: return Fake if prob > 0.5 else Real
        return "Fake" if prob > 0.5 else "Real", float(prob)

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return f"Error: {str(e)}", 0.0

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.mp4'):
            # Save uploaded file
            upload_path = os.path.join('uploads', file.filename)
            file.save(upload_path)
            
            # Predict and clean up
            result, confidence = predict_audio(upload_path)
            os.unlink(upload_path)
            
            return render_template('result.html', 
                                result=result,
                                confidence=round(confidence*100, 2))
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)