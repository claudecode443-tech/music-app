import numpy as np
import librosa
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import make_path_strictly_monotonic

FEATURE_RATE = 50       # synctoolbox default: 50 frames per second
SR_SYNCTOOLBOX = 22050  # synctoolbox filterbank is designed for 22 050 Hz


def align_score_to_audio(
    piano_roll: np.ndarray,
    audio_path: str,
    feature_rate: int = FEATURE_RATE,
) -> np.ndarray:
    """
    Align a score piano roll to audio using multi-scale MrMsDTW.

    Args:
        piano_roll: shape (n_score_frames, 128) at ``feature_rate`` fps.
        audio_path: path to the audio file (any format librosa can load).
        feature_rate: frames per second — must match the rate used to build
                      the piano roll (default 50, i.e. synctoolbox default).

    Returns:
        piano_roll_aligned: shape (n_audio_frames, 128) at ``feature_rate``
        fps, warped so that frame i corresponds to audio frame i.
    """
    audio, _ = librosa.load(audio_path, sr=SR_SYNCTOOLBOX, mono=True)

    # --- Score chroma (CENS) ---
    # pitch_to_chroma expects (128, N); piano_roll is (N, 128)
    piano_roll_T = piano_roll.T.astype(np.float64)            # (128, n_score_frames)
    chroma_score = pitch_to_chroma(f_pitch=piano_roll_T)      # (12, n_score_frames)
    chroma_score = quantize_chroma(chroma_score)

    # --- Audio chroma (CENS) ---
    f_pitch_audio = audio_to_pitch_features(
        f_audio=audio,
        Fs=SR_SYNCTOOLBOX,
        feature_rate=feature_rate,
        midi_min=21,
        midi_max=108,
    )  # (128, n_audio_frames)
    chroma_audio = pitch_to_chroma(f_pitch=f_pitch_audio)     # (12, n_audio_frames)
    chroma_audio = quantize_chroma(chroma_audio)

    n_score_frames = piano_roll.shape[0]
    n_audio_frames = chroma_audio.shape[1]

    # --- MrMsDTW ---
    # f_chroma1 = score (reference), f_chroma2 = audio (query)
    # Reversing these produces garbage alignment.
    alignment = sync_via_mrmsdtw(
        f_chroma1=chroma_score,
        f_chroma2=chroma_audio,
        input_feature_rate=feature_rate,
        verbose=False,
    )  # (2, T): alignment[0] = score frame indices, alignment[1] = audio frame indices

    alignment = make_path_strictly_monotonic(alignment)

    # --- Warp piano roll from score-frame space to audio-frame space ---
    all_audio_frames = np.arange(n_audio_frames, dtype=np.float64)
    score_frames_for_audio = np.interp(
        all_audio_frames,
        alignment[1].astype(np.float64),
        alignment[0].astype(np.float64),
    )
    score_frames_for_audio = np.clip(
        np.round(score_frames_for_audio).astype(np.int64),
        0,
        n_score_frames - 1,
    )

    piano_roll_aligned = piano_roll[score_frames_for_audio]   # (n_audio_frames, 128)
    return piano_roll_aligned
