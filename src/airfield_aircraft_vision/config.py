"""Small runtime configuration contract; provider secrets stay in adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    log_level: str = "INFO"
    cache_dir: str = "./.cache"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            log_level=os.getenv("AIRFIELD_VISION_LOG_LEVEL", "INFO"),
            cache_dir=os.getenv("AIRFIELD_VISION_CACHE_DIR", "./.cache"),
        )

