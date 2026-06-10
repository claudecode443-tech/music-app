from flask import Flask, request, send_file
import librosa
import numpy as np
import soundfile as sf
import music21
import os

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>Music Part Isolator</h1>
    <form action="/isolate" method="post" enctype="multipart/form-data">
        <p>Upload audio file (MP3):</p>
        <input type="file" name="audio" accept=".mp3"><br><br>
        <p>Upload score file (MusicXML):</p>
        <input type="file" name="score" accept=".xml,.musicxml"><br><br>
        <p>Select voice part:</p>
        <select name="part">
            <option value="0">Soprano</option>
            <option value="1">Alto</option>
            <option value="2">Tenor</option>
            <option value="3">Bass</option>
        </select><br><br>
        <input type="submit" value="Isolate My Part">
    </form>
    '''

@app.route('/isolate', methods=['POST'])
def isolate():
    audio_file = request.files['audio']
    score_file = request.files['score']
    part_index = int(request.form['part'])

    audio_path = 'uploaded_audio.mp3'
    score_path = 'uploaded_score.xml'
    audio_file.save(audio_path)
    score_file.save(score_path)

    print("Loading audio...")
    y, sr = librosa.load(audio_path, duration=30)

    print("Loading score...")
    score = music21.converter.parse(score_path)
    parts = score.parts
    
    if part_index >= len(parts):
        return f"Error: this score only has {len(parts)} parts"
    
    selected_part = parts[part_index]
    part_name = selected_part.partName or f"Part {part_index + 1}"
    print(f"Isolating: {part_name}")

    frequencies = []
    for note in selected_part.flatten().notes:
        frequencies.append(note.pitch.frequency)

    if not frequencies:
        return "Error: no notes found in selected part"

    print(f"Found {len(frequencies)} notes, filtering audio...")
    D = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)

    mask = np.zeros(len(freqs))
    for freq in frequencies:
        low, high = freq * 0.94, freq * 1.06
        mask += ((freqs >= low) & (freqs <= high)).astype(float)

    mask = np.clip(mask, 0, 1)
    D_filtered = D * mask[:, np.newaxis]
    y_filtered = librosa.istft(D_filtered)

    output_path = 'isolated_part.wav'
    sf.write(output_path, y_filtered, sr)
    print("Done!")

    return send_file(output_path, as_attachment=True, download_name=f'{part_name}_isolated.wav')

if __name__ == '__main__':
    app.run(debug=True)