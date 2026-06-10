import librosa
import numpy as np

# Load your audio file
audio_file = r"C:\Users\finla\Downloads\alexgrohl-classical-446258.mp3"

print("Loading audio file...")
y, sr = librosa.load(audio_file, duration=10)

print(f"Sample rate: {sr}")
print(f"Audio length: {len(y)} samples")
print(f"Duration: {round(len(y)/sr, 1)} seconds")
print("Success - librosa can read your audio file!")

import matplotlib.pyplot as plt

# Create a picture of the frequencies in the audio
D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(12, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Frequencies in your audio over time')
plt.tight_layout()
plt.savefig('audio_visual.png')
print("Image saved as audio_visual.png")