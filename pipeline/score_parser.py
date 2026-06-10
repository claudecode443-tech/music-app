import os
import shutil
import tempfile
import numpy as np
import music21
from music21 import tempo as m21tempo

FEATURE_RATE = 50  # synctoolbox default: 50 frames per second

ZIP_MAGIC = b"PK\x03\x04"


def _safe_parse(score_path: str):
    """
    Parse a MusicXML or MXL file.
    If the file has a .xml extension but contains ZIP/MXL bytes, copy it to a
    .mxl temp file so music21 uses the correct sub-converter.
    """
    with open(score_path, "rb") as f:
        header = f.read(4)

    ext = os.path.splitext(score_path)[1].lower()
    if header == ZIP_MAGIC and ext != ".mxl":
        tmp = tempfile.NamedTemporaryFile(suffix=".mxl", delete=False)
        tmp.close()
        shutil.copy2(score_path, tmp.name)
        try:
            return music21.converter.parse(tmp.name)
        finally:
            os.unlink(tmp.name)

    return music21.converter.parse(score_path)


def parse_score(score_path: str) -> list[str]:
    """Return list of part names from a MusicXML file."""
    score = _safe_parse(score_path)
    names = []
    for i, part in enumerate(score.parts):
        name = part.partName or part.id or f"Part {i + 1}"
        names.append(name)
    return names


def extract_piano_roll(
    score_path: str,
    part_name: str,
    feature_rate: int = FEATURE_RATE,
) -> np.ndarray:
    """
    Parse a MusicXML score and build a piano roll for the requested part.

    Returns an ndarray of shape (n_frames, 128) at the given feature_rate (fps).
    Each entry is 1.0 if that MIDI pitch is active at that frame, else 0.0.
    """
    score = _safe_parse(score_path)

    selected_part = None
    for i, part in enumerate(score.parts):
        name = part.partName or part.id or f"Part {i + 1}"
        if name == part_name:
            selected_part = part
            break

    if selected_part is None:
        available = parse_score(score_path)
        raise ValueError(
            f"Part '{part_name}' not found in score. "
            f"Available parts: {available}"
        )

    mm = score.recurse().getElementsByClass(m21tempo.MetronomeMark).first()
    bpm: float = 120.0
    if mm is not None:
        bpm_val = mm.getQuarterBPM()
        if bpm_val is not None:
            bpm = float(bpm_val)

    total_quarters = float(score.highestTime)
    total_seconds = total_quarters / bpm * 60.0
    n_frames = int(np.ceil(total_seconds * feature_rate)) + 1

    piano_roll = np.zeros((n_frames, 128), dtype=np.float32)

    for el in selected_part.flatten().notesAndRests:
        if el.isRest:
            continue
        onset_q = float(el.offset)
        dur_q = float(el.duration.quarterLength)
        if dur_q <= 0:
            continue
        onset_s = onset_q / bpm * 60.0
        offset_s = (onset_q + dur_q) / bpm * 60.0
        start_f = int(onset_s * feature_rate)
        end_f = min(int(offset_s * feature_rate), n_frames)
        if start_f >= end_f:
            end_f = start_f + 1

        if hasattr(el, "pitch"):
            midi = el.pitch.midi
            if 0 <= midi < 128:
                piano_roll[start_f:end_f, midi] = 1.0
        elif hasattr(el, "notes"):
            for note in el.notes:
                midi = note.pitch.midi
                if 0 <= midi < 128:
                    piano_roll[start_f:end_f, midi] = 1.0

    if piano_roll.sum() == 0:
        raise ValueError(
            f"No notes found in part '{part_name}'. "
            "Check that the part name exactly matches the score."
        )

    return piano_roll
