import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import respx
import soundfile as sf
from httpx import Response

from client import VoxCPMClient
from client.client import VoxCPMError


BASE = "http://localhost:8000"


def _make_wav_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.wav"
    data = np.zeros(8000, dtype=np.float32)
    sf.write(str(p), data, 16000)
    return p


@pytest.fixture
def tmp_wav(tmp_path):
    return _make_wav_file(tmp_path)


@respx.mock
def test_create_profile(tmp_wav):
    respx.post(f"{BASE}/profiles").mock(
        return_value=Response(201, json={
            "id": "abc123", "name": "Test", "description": None,
            "style_guidance": None, "tags": None,
            "created_at": "2026-01-01T00:00:00Z",
            "audio_format": "wav", "sample_rate": 16000,
            "duration_seconds": 0.5,
        })
    )
    with VoxCPMClient(BASE) as c:
        meta = c.create_profile(tmp_wav, "Test")
    assert meta["id"] == "abc123"


@respx.mock
def test_list_profiles():
    respx.get(f"{BASE}/profiles").mock(return_value=Response(200, json=[]))
    with VoxCPMClient(BASE) as c:
        assert c.list_profiles() == []


@respx.mock
def test_get_profile():
    respx.get(f"{BASE}/profiles/abc").mock(
        return_value=Response(200, json={"id": "abc", "name": "X",
            "description": None, "style_guidance": None, "tags": None,
            "created_at": "2026-01-01T00:00:00Z",
            "audio_format": "wav", "sample_rate": 16000,
            "duration_seconds": 1.0})
    )
    with VoxCPMClient(BASE) as c:
        assert c.get_profile("abc")["id"] == "abc"


@respx.mock
def test_get_profile_not_found():
    respx.get(f"{BASE}/profiles/missing").mock(
        return_value=Response(404, json={"detail": "Profile not found"})
    )
    with VoxCPMClient(BASE) as c:
        with pytest.raises(VoxCPMError) as exc_info:
            c.get_profile("missing")
        assert exc_info.value.status_code == 404


@respx.mock
def test_delete_profile():
    respx.delete(f"{BASE}/profiles/abc").mock(return_value=Response(204))
    with VoxCPMClient(BASE) as c:
        c.delete_profile("abc")  # Should not raise


@respx.mock
def test_generate(tmp_path):
    wav_bytes = b"RIFF" + b"\x00" * 100
    respx.post(f"{BASE}/generate").mock(
        return_value=Response(200, content=wav_bytes,
                              headers={"content-type": "audio/wav"})
    )
    out = tmp_path / "out.wav"
    with VoxCPMClient(BASE) as c:
        result = c.generate("abc", "Hello", output_path=out)
    assert result == wav_bytes
    assert out.read_bytes() == wav_bytes


@respx.mock
def test_health():
    respx.get(f"{BASE}/health").mock(
        return_value=Response(200, json={"status": "ok"})
    )
    with VoxCPMClient(BASE) as c:
        assert c.health() == {"status": "ok"}


@respx.mock
def test_ready_true():
    respx.get(f"{BASE}/ready").mock(
        return_value=Response(200, json={"status": "ready"})
    )
    with VoxCPMClient(BASE) as c:
        assert c.ready() is True


@respx.mock
def test_ready_false():
    respx.get(f"{BASE}/ready").mock(return_value=Response(503))
    with VoxCPMClient(BASE) as c:
        assert c.ready() is False
