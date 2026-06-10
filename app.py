from flask import Flask, request, send_file, redirect, url_for
import librosa
import numpy as np
import soundfile as sf
import music21
import requests
import os
import subprocess
import glob

app = Flask(__name__)

def get_api_key():
    key = os.environ.get('AUDD_API_KEY', '')
    if not key:
        try:
            with open('.env') as f:
                for line in f:
                    if line.startswith('AUDD_API_KEY='):
                        key = line.strip().split('=', 1)[1]
        except FileNotFoundError:
            pass
    return key

def identify_piece(audio_path):
    api_key = get_api_key()
    import soundfile as sf
    import tempfile
    y, sr = librosa.load(audio_path, offset=10, duration=15)
    temp_path = 'temp_identify.wav'
    sf.write(temp_path, y, sr)
    with open(temp_path, 'rb') as f:
        result = requests.post(
            'https://api.audd.io/',
            data={'api_token': api_key},
            files={'file': f}
        )

    data = result.json()
    if data.get('status') == 'success' and data.get('result'):
        piece = data['result']
        return piece.get('title', 'Unknown'), piece.get('artist', 'Unknown')
    return None, None

def separate_vocals(audio_path):
    print("Separating vocals with Demucs (this takes 1-2 minutes)...")
    subprocess.run([
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "--out", "separated",
        audio_path
    ], capture_output=True)

    vocals_files = glob.glob("separated/**/*.wav", recursive=True)
    vocals_path = next((f for f in vocals_files if 'vocals' in f.lower()), None)
    return vocals_path

def filter_by_score(audio_path, score_path, part_index):
    y, sr = librosa.load(audio_path, duration=60)
    score = music21.converter.parse(score_path)
    parts = score.parts

    if part_index >= len(parts):
        return None, f"Score only has {len(parts)} parts"

    selected_part = parts[part_index]
    part_name = selected_part.partName or f"Part {part_index + 1}"

    frequencies = [note.pitch.frequency for note in selected_part.flatten().notes]
    if not frequencies:
        return None, "No notes found in selected part"

    D = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)

    mask = np.zeros(len(freqs))
    for freq in frequencies:
        low, high = freq * 0.90, freq * 1.10
        mask += ((freqs >= low) & (freqs <= high)).astype(float)

    mask = np.clip(mask, 0, 1)
    D_filtered = D * mask[:, np.newaxis]
    y_filtered = librosa.istft(D_filtered)

    output_path = 'isolated_part.wav'
    sf.write(output_path, y_filtered, sr)
    return output_path, part_name

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Music Part Isolator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #2c3e50; }
            .card { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            label { font-weight: bold; display: block; margin-top: 12px; }
            input[type=file], select { width: 100%; padding: 8px; margin: 8px 0 16px 0; border: 1px solid #ddd; border-radius: 4px; }
            input[type=submit] { background: #2c3e50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            input[type=submit]:hover { background: #34495e; }
            p { color: #666; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Music Part Isolator</h1>
            <p>Upload your recording and score to isolate your voice part.</p>
            <form action="/isolate" method="post" enctype="multipart/form-data">
                <label>Audio recording (MP3):</label>
                <input type="file" name="audio" accept=".mp3" required>
                <label>Score file (MusicXML or .mxl):</label>
                <input type="file" name="score" accept=".xml,.musicxml,.mxl" required>
                <label>Your voice part:</label>
                <select name="part">
                    <option value="0">Soprano</option>
                    <option value="1">Alto</option>
                    <option value="2">Tenor</option>
                    <option value="3">Bass</option>
                </select>
                <br>
                <input type="submit" value="Isolate My Part →">
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/isolate', methods=['POST'])
def isolate():
    audio_file = request.files.get('audio')
    score_file = request.files['score']
    part_index = int(request.form['part'])
    title = request.form.get('title', 'piece')

    if audio_file:
        audio_file.save('uploaded_audio.mp3')

    ext = os.path.splitext(score_file.filename)[1].lower()
    score_path = f'uploaded_score{ext}'
    score_file.save(score_path)

    print("Separating vocals...")
    vocals_path = separate_vocals('uploaded_audio.mp3')

    audio_to_filter = vocals_path if vocals_path else 'uploaded_audio.mp3'
    if vocals_path:
        print("Using Demucs vocal stem for cleaner result")
    else:
        print("Demucs separation failed, using original audio")

    print("Applying score-based filtering...")
    output_path, part_name = filter_by_score(audio_to_filter, score_path, part_index)

    if not output_path:
        return f"<h2>Error: {part_name}</h2><a href='/'>Try again</a>"

    print("Done!")
    return send_file(output_path, as_attachment=True, download_name=f'{title}_{part_name}.wav')

if __name__ == '__main__':
    app.run(debug=True)
