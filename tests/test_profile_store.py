import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from server.profile_store import ProfileStore


@pytest_asyncio.fixture
async def store(tmp_path):
    return ProfileStore(tmp_path)


@pytest.mark.asyncio
async def test_create_and_get(store):
    meta = await store.create(
        name="test_voice",
        audio_bytes=b"fake_audio_data",
        audio_format="wav",
        sample_rate=48000,
        duration_seconds=3.5,
        description="A test voice",
        style_guidance="calm and slow",
        tags=["test", "english"],
    )
    assert meta.name == "test_voice"
    assert meta.sample_rate == 48000

    fetched = await store.get(meta.id)
    assert fetched is not None
    assert fetched.id == meta.id
    assert fetched.description == "A test voice"


@pytest.mark.asyncio
async def test_save_and_get_latents(store):
    meta = await store.create(
        name="v", audio_bytes=b"x", audio_format="wav",
        sample_rate=48000, duration_seconds=1.0,
    )
    await store.save_latents(meta.id, b"latent_data_here")
    latents = await store.get_latents(meta.id)
    assert latents == b"latent_data_here"


@pytest.mark.asyncio
async def test_list(store):
    await store.create(name="a", audio_bytes=b"x", audio_format="wav", sample_rate=48000, duration_seconds=1.0)
    await store.create(name="b", audio_bytes=b"y", audio_format="wav", sample_rate=48000, duration_seconds=2.0)
    profiles = await store.list()
    assert len(profiles) == 2
    names = {p.name for p in profiles}
    assert names == {"a", "b"}


@pytest.mark.asyncio
async def test_delete(store):
    meta = await store.create(name="del", audio_bytes=b"x", audio_format="wav", sample_rate=48000, duration_seconds=1.0)
    assert await store.delete(meta.id) is True
    assert await store.get(meta.id) is None
    assert await store.delete(meta.id) is False


@pytest.mark.asyncio
async def test_get_nonexistent(store):
    assert await store.get("nonexistent") is None
    assert await store.get_latents("nonexistent") is None
