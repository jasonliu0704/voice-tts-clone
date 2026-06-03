from datetime import datetime
from pydantic import BaseModel


class ProfileMetadata(BaseModel):
    id: str
    name: str
    description: str | None = None
    style_guidance: str | None = None
    tags: list[str] | None = None
    created_at: datetime
    audio_format: str
    sample_rate: int
    duration_seconds: float


class GenerateRequest(BaseModel):
    profile_id: str
    target_text: str
    style_guidance: str | None = None
    cfg_value: float = 2.0
    temperature: float = 1.0
    max_generate_length: int = 2000
