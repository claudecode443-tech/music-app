# PartIsolator

Score-informed classical music part isolation. Upload a recording and a MusicXML score; get back a WAV of just your voice or instrument part.

## How it works

1. **Demucs** separates vocals from accompaniment
2. **music21** parses the MusicXML score into a time-stamped piano roll
3. **synctoolbox MrMsDTW** aligns the score timeline to the audio (handles tempo drift)
4. A **time-varying Wiener soft mask** is applied to the vocal STFT — frequency bands open only when the score says that note is being sung, producing natural-sounding output

## Setup

### Requirements

- Python 3.11+
- [Audiveris](https://github.com/Audiveris/audiveris) (optional — for PDF input, not yet enabled)

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://localhost:8000](http://localhost:8000).

## Usage

1. Upload an MP3/WAV recording
2. Upload a MusicXML score (`.xml` or `.mxl`) — parts are detected automatically
3. Select your part from the dropdown
4. Click **Isolate My Part** and wait 1–4 minutes (Demucs is slow on CPU)
5. Download the isolated WAV

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/parts` | Multipart: `score` file → `{"parts": [...]}` |
| `POST` | `/isolate` | Multipart: `audio`, `score`, `part_name` → `{"job_id": "..."}` |
| `GET`  | `/status/{job_id}` | `{"status", "pct", "msg", "error"}` |
| `GET`  | `/download/{job_id}` | WAV file download |

## File structure

```
main.py              FastAPI app
pipeline/
  score_parser.py    music21 parsing + piano roll
  separator.py       Demucs subprocess
  aligner.py         synctoolbox DTW alignment
  filter.py          time-varying Wiener mask
  pdf_converter.py   stub (PDF not yet supported)
static/
  index.html         PWA frontend
  manifest.json      PWA manifest
  sw.js              Service worker
```

## Performance notes

- **CPU-only**: Demucs runs at ~3–4× real-time. A 10-minute file takes 2.5–4 min.
- **CUDA**: If `torch.cuda.is_available()`, Demucs uses GPU automatically and is 10–20× faster.
- Processing target: < 3 minutes for files under ~5 minutes long on GPU.

## Limitations (MVP)

- PDF score input not yet supported — use MusicXML
- No user accounts or persistent history
- Stripe integration not wired up (pricing UI is placeholder)
- Rate limit: 3 free isolations/month per IP (in-memory, resets on server restart)
