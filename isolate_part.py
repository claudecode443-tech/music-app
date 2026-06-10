import librosa
import numpy as np
import soundfile as sf
import music21

# Load audio
print("Loading audio...")
audio_file = r"C:\Users\finla\Downloads\alexgrohl-classical-446258.mp3"
y, sr = librosa.load(audio_file, duration=30)

# Load score and get Bass frequencies
print("Loading score...")
score = music21.corpus.parse('bach/bwv66.6')
bass_part = score.parts[3]  # Bass is part 4

bass_frequencies = []
for note in bass_part.flatten().notes:
    bass_frequencies.append(note.pitch.frequency)

print(f"Bass has {len(bass_frequencies)} notes")
print(f"Frequency range: {round(min(bass_frequencies))} Hz to {round(max(bass_frequencies))} Hz")

# Filter audio to only keep Bass frequencies
print("Filtering audio...")
D = librosa.stft(y)
frequencies = librosa.fft_frequencies(sr=sr)

mask = np.zeros(len(frequencies))
for freq in bass_frequencies:
    low, high = freq * 0.94, freq * 1.06
    mask += ((frequencies >= low) & (frequencies <= high)).astype(float)

mask = np.clip(mask, 0, 1)
D_filtered = D * mask[:, np.newaxis]

# Save the result
print("Saving filtered audio...")
y_filtered = librosa.istft(D_filtered)
sf.write('bass_only.wav', y_filtered, sr)
print("Done! Saved as bass_only.wav")