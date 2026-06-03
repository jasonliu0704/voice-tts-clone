import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import ProfileMetadata


class ProfileStore:
    def __init__(self, data_dir: Path):
        self._profiles_dir = data_dir / "profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

    def _profile_dir(self, profile_id: str) -> Path:
        return self._profiles_dir / profile_id

    async def create(
        self,
        name: str,
        audio_bytes: bytes,
        audio_format: str,
        sample_rate: int,
        duration_seconds: float,
        description: str | None = None,
        style_guidance: str | None = None,
        tags: list[str] | None = None,
    ) -> ProfileMetadata:
        profile_id = uuid.uuid4().hex
        profile_dir = self._profile_dir(profile_id)
        profile_dir.mkdir(parents=True)

        # Write audio
        (profile_dir / f"audio.{audio_format}").write_bytes(audio_bytes)

        meta = ProfileMetadata(
            id=profile_id,
            name=name,
            description=description,
            style_guidance=style_guidance,
            tags=tags,
            created_at=datetime.now(timezone.utc),
            audio_format=audio_format,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
        (profile_dir / "metadata.json").write_text(meta.model_dump_json())
        return meta

    async def save_latents(self, profile_id: str, latents: bytes) -> None:
        (self._profile_dir(profile_id) / "latents.bin").write_bytes(latents)

    async def get(self, profile_id: str) -> ProfileMetadata | None:
        meta_path = self._profile_dir(profile_id) / "metadata.json"
        if not meta_path.exists():
            return None
        return ProfileMetadata.model_validate_json(meta_path.read_text())

    async def get_latents(self, profile_id: str) -> bytes | None:
        latents_path = self._profile_dir(profile_id) / "latents.bin"
        if not latents_path.exists():
            return None
        return latents_path.read_bytes()

    async def list(self) -> list[ProfileMetadata]:
        profiles = []
        for meta_path in self._profiles_dir.glob("*/metadata.json"):
            profiles.append(ProfileMetadata.model_validate_json(meta_path.read_text()))
        return profiles

    async def delete(self, profile_id: str) -> bool:
        profile_dir = self._profile_dir(profile_id)
        if not profile_dir.exists():
            return False
        shutil.rmtree(profile_dir)
        return True
