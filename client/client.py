"""VoxCPM Voice Profile Server — Python Client SDK.

Provides a synchronous HTTP client for the VoxCPM Voice Profile Server.
Supports profile CRUD and speech generation with voice cloning.

The server uses VoxCPM-0.5B (16kHz output). The ``description`` field on a
profile is used as the reference transcript for voice cloning — provide the
actual spoken content of the reference audio for best results.

Example::

    from client import VoxCPMClient

    with VoxCPMClient("http://localhost:8001") as c:
        profile = c.create_profile(
            "reference.wav",
            name="Alice",
            description="The words spoken in reference.wav",
        )
        c.generate(profile["id"], "Hello world!", output_path="out.wav")
"""

from pathlib import Path

import httpx


class VoxCPMError(Exception):
    """Raised when the server returns an HTTP 4xx/5xx response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class VoxCPMClient:
    """Synchronous client for the VoxCPM Voice Profile Server.

    Args:
        base_url: Server base URL (e.g. ``http://localhost:8001``).
        timeout: Request timeout in seconds. Generation can be slow;
            default is 120s.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _raise(self, r: httpx.Response) -> None:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise VoxCPMError(r.status_code, detail)

    def create_profile(
        self,
        audio_path: str | Path,
        name: str,
        *,
        description: str | None = None,
        style_guidance: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Upload reference audio and create a voice profile.

        Args:
            audio_path: Path to reference audio file (WAV, FLAC, MP3, OGG).
            name: Display name for the profile.
            description: Transcript of the reference audio. Used as
                ``prompt_text`` during generation for voice cloning accuracy.
                Provide the actual words spoken in the audio for best results.
            style_guidance: Default style guidance for this voice.
            tags: Optional list of tags for organization.

        Returns:
            Profile metadata dict with ``id``, ``name``, ``description``,
            ``style_guidance``, ``tags``, ``created_at``, ``audio_format``,
            ``sample_rate``, ``duration_seconds``.
        """
        audio_path = Path(audio_path)
        files = {"audio": (audio_path.name, audio_path.read_bytes())}
        data: dict = {"name": name}
        if description:
            data["description"] = description
        if style_guidance:
            data["style_guidance"] = style_guidance
        if tags:
            data["tags"] = ",".join(tags)
        r = self._client.post("/profiles", files=files, data=data)
        self._raise(r)
        return r.json()

    def list_profiles(self) -> list[dict]:
        """List all stored voice profiles."""
        r = self._client.get("/profiles")
        self._raise(r)
        return r.json()

    def get_profile(self, profile_id: str) -> dict:
        """Get metadata for a specific profile.

        Raises:
            VoxCPMError: 404 if profile not found.
        """
        r = self._client.get(f"/profiles/{profile_id}")
        self._raise(r)
        return r.json()

    def delete_profile(self, profile_id: str) -> None:
        """Delete a voice profile.

        Raises:
            VoxCPMError: 404 if profile not found.
        """
        r = self._client.delete(f"/profiles/{profile_id}")
        self._raise(r)

    def generate(
        self,
        profile_id: str,
        text: str,
        *,
        style_guidance: str | None = None,
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        output_path: str | Path | None = None,
    ) -> bytes:
        """Generate speech using a stored voice profile.

        Args:
            profile_id: ID of the voice profile to clone.
            text: Text to synthesize.
            style_guidance: Optional style override (not used by VoxCPM-0.5B).
            cfg_value: Classifier-free guidance strength (higher = closer
                to reference voice, but may reduce quality). Default 2.0.
            temperature: Sampling temperature. Default 1.0.
            max_generate_length: Max generation length in tokens.
            output_path: If provided, saves the WAV file to this path.

        Returns:
            Raw WAV bytes (16kHz, mono).

        Raises:
            VoxCPMError: 404 if profile not found, 422 if latents missing,
                503 if engine not ready.
        """
        payload: dict = {
            "profile_id": profile_id,
            "target_text": text,
            "cfg_value": cfg_value,
            "temperature": temperature,
            "max_generate_length": max_generate_length,
        }
        if style_guidance:
            payload["style_guidance"] = style_guidance
        r = self._client.post("/generate", json=payload)
        self._raise(r)
        if output_path:
            Path(output_path).write_bytes(r.content)
        return r.content

    def health(self) -> dict:
        """Check server liveness. Returns ``{"status": "ok"}``."""
        r = self._client.get("/health")
        self._raise(r)
        return r.json()

    def ready(self) -> bool:
        """Check if the TTS engine is loaded and ready for generation."""
        r = self._client.get("/ready")
        return r.status_code == 200

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
