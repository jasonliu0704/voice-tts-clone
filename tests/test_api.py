import io
import numpy as np
import soundfile as sf
import pytest
from httpx import ASGITransport, AsyncClient

from server.main import app
from server.profile_store import ProfileStore


def _make_wav() -> bytes:
    """Generate a short sine wave WAV file."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
    data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return _make_wav()


@pytest.fixture
async def client(tmp_path):
    # Override data dir to use temp path for tests
    from server.config import settings
    settings.voxcpm_data_dir = str(tmp_path)
    app.state.store = ProfileStore(tmp_path)
    app.state.engine = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_create_and_get_profile(client, wav_bytes):
    r = await client.post(
        "/profiles",
        files={"audio": ("voice.wav", wav_bytes, "audio/wav")},
        data={"name": "TestVoice", "description": "A test", "tags": "en,test"},
    )
    assert r.status_code == 201
    meta = r.json()
    assert meta["name"] == "TestVoice"
    assert meta["description"] == "A test"
    assert meta["tags"] == ["en", "test"]
    assert meta["sample_rate"] == 16000
    profile_id = meta["id"]

    # GET single
    r = await client.get(f"/profiles/{profile_id}")
    assert r.status_code == 200
    assert r.json()["id"] == profile_id


async def test_list_profiles(client, wav_bytes):
    await client.post(
        "/profiles",
        files={"audio": ("a.wav", wav_bytes, "audio/wav")},
        data={"name": "A"},
    )
    r = await client.get("/profiles")
    assert r.status_code == 200
    assert len(r.json()) >= 1


async def test_delete_profile(client, wav_bytes):
    r = await client.post(
        "/profiles",
        files={"audio": ("d.wav", wav_bytes, "audio/wav")},
        data={"name": "Del"},
    )
    pid = r.json()["id"]
    r = await client.delete(f"/profiles/{pid}")
    assert r.status_code == 204
    r = await client.get(f"/profiles/{pid}")
    assert r.status_code == 404


async def test_get_nonexistent(client):
    r = await client.get("/profiles/nonexistent")
    assert r.status_code == 404


async def test_ready_no_engine(client):
    r = await client.get("/ready")
    assert r.status_code == 503


async def test_generate_no_engine(client, wav_bytes):
    # Create a profile first
    r = await client.post(
        "/profiles",
        files={"audio": ("v.wav", wav_bytes, "audio/wav")},
        data={"name": "V"},
    )
    pid = r.json()["id"]
    # Generate without engine should 503
    r = await client.post("/generate", json={"profile_id": pid, "target_text": "hello"})
    assert r.status_code == 503


async def test_generate_with_mock_engine(tmp_path, wav_bytes):
    """Test /generate with a fake engine that returns a sine wave."""
    import numpy as np
    from unittest.mock import AsyncMock

    app.state.store = ProfileStore(tmp_path)

    # Create a mock engine
    mock_engine = AsyncMock()
    mock_engine.is_ready = True
    mock_engine.sample_rate = 16000
    mock_engine.generate = AsyncMock(
        return_value=np.zeros(8000, dtype=np.float32)
    )
    mock_engine.encode_latents = AsyncMock(return_value=b"\x00" * 128)
    app.state.engine = mock_engine

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create profile
        r = await client.post(
            "/profiles",
            files={"audio": ("v.wav", wav_bytes, "audio/wav")},
            data={"name": "MockVoice"},
        )
        assert r.status_code == 201
        pid = r.json()["id"]

        # Generate
        r = await client.post(
            "/generate",
            json={"profile_id": pid, "target_text": "Hello world"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"
        assert len(r.content) > 44  # WAV header is 44 bytes
        mock_engine.generate.assert_called_once()
