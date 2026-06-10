import numpy as np
import librosa
import soundfile as sf

N_FFT = 2048
HOP_LENGTH = 512
TOLERANCE = 1.15                       # ±15% bandwidth around each harmonic
HARMONICS = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)]
NOISE_FLOOR = 0.1                      # keeps inactive regions at ~-20dB, not silence
EPSILON = 1e-8
ALIGN_FEATURE_RATE = 50                # fps of the aligned piano roll coming in


def _build_midi_freq_weight_matrix(sr: int, n_fft: int) -> np.ndarray:
    """
    Precompute (128, n_freq_bins) weight matrix mapping MIDI pitch → STFT bins.

    Entry [midi, bin] = sum of harmonic weights that place energy in that bin.
    Computed once per job call; avoids a per-frame Python loop.
    """
    n_freq_bins = 1 + n_fft // 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    weights = np.zeros((128, n_freq_bins), dtype=np.float32)

    for midi in range(21, 109):
        f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        for h_mult, h_weight in HARMONICS:
            fh = f0 * h_mult
            if fh >= sr / 2:
                break
            low = fh / TOLERANCE
            high = fh * TOLERANCE
            active = (freqs >= low) & (freqs <= high)
            weights[midi, active] += h_weight

    return weights


def apply_score_mask(
    vocals_path: str,
    piano_roll_aligned: np.ndarray,
    output_path: str,
    sr: int = 22050,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> str:
    """
    Apply a time-varying Wiener soft mask to the vocal stem.

    The mask opens frequency bands only when the score says a note is active,
    producing natural-sounding output rather than the static-mask artefacts
    of the original prototype.

    Args:
        vocals_path: path to the Demucs vocal stem WAV.
        piano_roll_aligned: (n_align_frames, 128) float32 at ALIGN_FEATURE_RATE fps,
                            already warped to audio timing by aligner.py.
        output_path: where to write the output WAV.
        sr: sample rate to use throughout.
        n_fft: STFT window size.
        hop_length: STFT hop size.

    Returns:
        output_path
    """
    y, _ = librosa.load(vocals_path, sr=sr, mono=True)

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    n_stft_frames = D.shape[1]

    # --- Resample piano roll from align fps to STFT fps ---
    stft_fps = sr / hop_length                        # ~43.07 fps at 22050/512
    n_align_frames = piano_roll_aligned.shape[0]
    align_times = np.arange(n_align_frames, dtype=np.float64) / ALIGN_FEATURE_RATE
    stft_times = np.arange(n_stft_frames, dtype=np.float64) / stft_fps

    piano_roll_resampled = np.zeros((n_stft_frames, 128), dtype=np.float32)
    for midi in range(128):
        col = piano_roll_aligned[:, midi].astype(np.float64)
        if col.max() > 0:
            piano_roll_resampled[:, midi] = np.interp(stft_times, align_times, col).astype(np.float32)

    piano_roll_resampled = (piano_roll_resampled > 0.5).astype(np.float32)

    # --- Build frequency mask via matrix multiply (no Python loop per frame) ---
    midi_freq_weights = _build_midi_freq_weight_matrix(sr=sr, n_fft=n_fft)
    # (n_stft_frames, 128) @ (128, 1025) → (n_stft_frames, 1025)
    mask_raw = (piano_roll_resampled @ midi_freq_weights).T  # → (1025, n_stft_frames)

    # --- Wiener soft mask ---
    # mask_soft ≈ 1 where notes are active, ≈ 0 elsewhere
    # NOISE_FLOOR^2 in denominator prevents hard silence in inactive regions
    mask_sq = mask_raw ** 2
    soft_mask = mask_sq / (mask_sq + NOISE_FLOOR ** 2 + EPSILON)

    # --- Apply mask, preserving original phase ---
    D_masked = D * soft_mask

    # --- Reconstruct ---
    y_out = librosa.istft(D_masked, hop_length=hop_length, length=len(y))

    sf.write(output_path, y_out, sr)
    return output_path
