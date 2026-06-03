import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .models import GenerateRequest, ProfileMetadata
from .profile_store import ProfileStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ProfileStore(Path(settings.voxcpm_data_dir))
    app.state.engine = None
    try:
        from .engine import TTSEngine

        engine = TTSEngine()
        await engine.start(settings.voxcpm_model_path, settings.device_list)
        app.state.engine = engine
    except Exception:
        pass  # Engine unavailable (no GPU or missing package)
    yield
    if app.state.engine:
        await app.state.engine.stop()


app = FastAPI(title="VoxCPM Voice Profile Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not app.state.engine or not app.state.engine.is_ready:
        raise HTTPException(503, "Engine not ready")
    return {"status": "ready"}


@app.post("/profiles", status_code=201, response_model=ProfileMetadata)
async def create_profile(
    audio: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    style_guidance: str | None = Form(None),
    tags: str | None = Form(None),
):
    audio_bytes = await audio.read()

    # Determine format from filename
    ext = (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    if ext not in ("wav", "flac", "mp3", "ogg"):
        ext = "wav"

    # Read audio info
    buf = io.BytesIO(audio_bytes)
    try:
        data, sample_rate = sf.read(buf)
    except Exception:
        raise HTTPException(422, "Could not read audio file")
    duration = len(data) / sample_rate

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    store: ProfileStore = app.state.store
    meta = await store.create(
        name=name,
        audio_bytes=audio_bytes,
        audio_format=ext,
        sample_rate=sample_rate,
        duration_seconds=round(duration, 3),
        description=description,
        style_guidance=style_guidance,
        tags=tag_list,
    )

    # Pre-compute latents if engine available
    if app.state.engine:
        latents = await app.state.engine.encode_latents(audio_bytes, ext)
        await store.save_latents(meta.id, latents)

    return meta


@app.get("/profiles", response_model=list[ProfileMetadata])
async def list_profiles():
    return await app.state.store.list()


@app.get("/profiles/{profile_id}", response_model=ProfileMetadata)
async def get_profile(profile_id: str):
    meta = await app.state.store.get(profile_id)
    if not meta:
        raise HTTPException(404, "Profile not found")
    return meta


@app.delete("/profiles/{profile_id}", status_code=204)
async def delete_profile(profile_id: str):
    if not await app.state.store.delete(profile_id):
        raise HTTPException(404, "Profile not found")


@app.post("/generate")
async def generate(req: GenerateRequest):
    engine = app.state.engine
    if not engine or not engine.is_ready:
        raise HTTPException(503, "Engine not ready")

    store: ProfileStore = app.state.store
    meta = await store.get(req.profile_id)
    if not meta:
        raise HTTPException(404, "Profile not found")

    latents = await store.get_latents(req.profile_id)
    if not latents:
        raise HTTPException(422, "Profile latents not computed")

    wav_array = await engine.generate(
        target_text=req.target_text,
        ref_audio_latents=latents,
        cfg_value=req.cfg_value,
        temperature=req.temperature,
        max_generate_length=req.max_generate_length,
        prompt_text=meta.description or meta.style_guidance or ".",
    )

    # Encode as WAV
    buf = io.BytesIO()
    sf.write(buf, wav_array, engine.sample_rate, format="WAV")
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Audio-Sample-Rate": str(engine.sample_rate)},
    )
