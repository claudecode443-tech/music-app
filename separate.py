import subprocess
import os

audio_path = r"C:\Users\finla\Downloads\Cantata ''Jesu, der du meine Seele'', BWV. 78 - 2. Wir eilen mit schwachen (1).mp3"

print("Separating audio into stems...")
subprocess.run([
    "python", "-m", "demucs",
    "--two-stems", "vocals",
    audio_path
])

print("Done! Check the 'separated' folder")