import os
import numpy as np
import torch
import librosa
import soundfile as sf

MODEL_NAME = "htdemucs"
TARGET_SR = 44100  # htdemucs native sample rate


def separate_vocals(audio_path: str, output_dir: str) -> str:
    """
    Separate vocals from a mixed audio file using Demucs (Python API).

    Loads audio with librosa (avoids torchcodec dependency), runs the
    htdemucs model, writes vocals.wav to output_dir.

    Returns the path to vocals.wav.
    Raises RuntimeError on failure.
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Load audio with librosa — stereo, 44100 Hz (htdemucs requirement)
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)  # mono → fake stereo

    # (channels, samples) → (1, channels, samples) batch
    wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)

    model = get_model(MODEL_NAME)
    model.eval()

    with torch.no_grad():
        sources = apply_model(model, wav, progress=False)[0]
        # sources: (n_sources, channels, samples)

    vocal_idx = model.sources.index("vocals")
    vocals_np = sources[vocal_idx].numpy()  # (channels, samples)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "vocals.wav")
    # soundfile expects (samples, channels)
    sf.write(out_path, vocals_np.T, TARGET_SR)

    return out_path
