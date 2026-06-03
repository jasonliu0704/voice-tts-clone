import asyncio
import io
import json
import tempfile
from functools import partial

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM


class TTSEngine:
    def __init__(self):
        self._model = None
        self._sample_rate: int = 16000
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self, model_path: str, devices: list[int]) -> None:
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, partial(VoxCPM.from_pretrained, model_path, device=f"cuda:{devices[0]}")
        )
        self._ready = True

    async def encode_latents(self, audio_bytes: bytes, audio_format: str) -> bytes:
        # Store raw audio bytes as-is (VoxCPM-0.5B doesn't pre-compute latents)
        return audio_bytes

    async def generate(
        self,
        target_text: str,
        ref_audio_latents: bytes,
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        prompt_text: str = ".",
    ) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            try:
                data, sr = sf.read(io.BytesIO(ref_audio_latents))
                sf.write(f.name, data, sr, format="WAV")
            except Exception:
                f.write(ref_audio_latents)
            prompt_path = f.name

        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(
            None,
            partial(
                self._model.generate,
                text=target_text,
                prompt_wav_path=prompt_path,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=10,
            ),
        )
        return np.asarray(wav, dtype=np.float32)

    async def stop(self) -> None:
        self._model = None
        self._ready = False
