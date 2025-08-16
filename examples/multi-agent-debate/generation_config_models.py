import json
from typing import Optional

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Represents configuration for text generation."""

    max_output_tokens: int = Field(
        default=1024, ge=1, description="Maximum output tokens."
    )
    min_output_tokens: int = Field(
        default=1, ge=0, description="Minimum output tokens."
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Sampling temperature."
    )
    seed: Optional[int] = Field(
        default=42,
        description="Seed for reproducibility. If set, ensures deterministic "
        "outputs for the same input.",
    )


def load_generation_config(file_path: str) -> GenerationConfig:
    """
    Load and validate generation configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        GenerationConfig: Validated generation configuration.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    return GenerationConfig(**config_data)
