# VoxCPM Voice Profile Server

A production API server for **VoxCPM2 controllable voice cloning** powered by [Nano-vLLM](https://github.com/a710128/nanovllm-voxcpm). Upload a reference voice once, store it as a reusable profile, and generate cloned speech on demand with optional style control.

## Features

- **Voice Profile CRUD** — Upload reference audio, get back a profile ID for repeated use
- **Controllable Voice Cloning** — Generate speech with stored voice + optional style guidance
- **Pre-computed Latents** — Reference audio is encoded once at upload time for fast generation
- **48kHz WAV Output** — Studio-quality audio output from VoxCPM2
- **Python Client SDK** — Clean SDK with context manager support

## Quick Start

### Install

```bash
pip install -e ".[gpu]"  # GPU machine with CUDA 12+
pip install -e ".[dev]"  # Development/testing only
```

### Run Server

```bash
export VOXCPM_MODEL_PATH=openbmb/VoxCPM2  # or local path
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### Client Usage

```python
from client import VoxCPMClient

with VoxCPMClient("http://localhost:8000") as c:
    # Upload a voice profile
    profile = c.create_profile(
        "reference_voice.wav",
        name="Alice",
        description="Female, mid-30s, calm",
        style_guidance="warm and friendly tone",
    )

    # Generate speech with that voice
    c.generate(
        profile["id"],
        "Hello! Welcome to our service.",
        output_path="output.wav",
    )

    # Override style at generation time
    c.generate(
        profile["id"],
        "This is urgent!",
        style_guidance="fast and energetic",
        output_path="urgent.wav",
    )
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness (model loaded) |
| `POST` | `/profiles` | Create profile (multipart: audio + metadata) |
| `GET` | `/profiles` | List all profiles |
| `GET` | `/profiles/{id}` | Get profile metadata |
| `DELETE` | `/profiles/{id}` | Delete profile |
| `POST` | `/generate` | Generate speech (JSON body, returns WAV) |

### POST /profiles

Multipart form fields:
- `audio` (file, required) — Reference audio (WAV, FLAC, MP3, OGG)
- `name` (string, required)
- `description` (string, optional)
- `style_guidance` (string, optional) — Default style for this voice
- `tags` (string, optional) — Comma-separated

### POST /generate

JSON body:
```json
{
  "profile_id": "abc123",
  "target_text": "Text to synthesize",
  "style_guidance": "calm and measured",
  "cfg_value": 2.0,
  "temperature": 1.0,
  "max_generate_length": 2000
}
```

Returns: `audio/wav` response body.

## Docker

```bash
# Build
docker compose build

# Run (mount your model weights)
VOXCPM_MODEL_PATH=/path/to/models docker compose up
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXCPM_MODEL_PATH` | `openbmb/VoxCPM2` | Model path or HuggingFace repo ID |
| `VOXCPM_DATA_DIR` | `./data` | Profile storage directory |
| `VOXCPM_DEVICES` | `0` | Comma-separated GPU device IDs |
| `VOXCPM_HOST` | `0.0.0.0` | Server bind host |
| `VOXCPM_PORT` | `8000` | Server bind port |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

```
Client SDK ──HTTP──▶ FastAPI Server ──▶ Profile Store (filesystem)
                           │
                           ▼
                    Nano-vLLM Engine (GPU)
                           │
                           ▼
                    VoxCPM2 Model (48kHz)
```

Each profile is stored at `DATA_DIR/profiles/<uuid>/`:
- `metadata.json` — Profile metadata
- `audio.<ext>` — Original reference audio
- `latents.bin` — Pre-computed latent representation
# voice-tts-clone
