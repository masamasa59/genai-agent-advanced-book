import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env を確実に読んで os.environ に入れる（ノートブックのカレントに依存しない）
_chapter4_dir = Path(__file__).resolve().parent.parent
_env_candidates = [
    _chapter4_dir / ".env",
    Path.cwd() / ".env",
    Path.cwd().parent / ".env",
    Path.cwd() / "chapter4" / ".env",
]
_env_path = next((p for p in _env_candidates if p.is_file()), _chapter4_dir / ".env")

if _env_path.is_file():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


class Settings(BaseSettings):
    openai_api_key: str
    openai_api_base: str
    openai_model: str

    model_config = SettingsConfigDict(
        env_file=str(_env_path),
        env_file_encoding="utf-8",
        extra="ignore",
    )
