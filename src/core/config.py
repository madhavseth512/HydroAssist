"""Configuration management for HydroAssist."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    persist_directory: str = "./data/vectordb"
    collection_name: str = "hydrogeology_kb"


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    provider: str = "gemini"
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    max_tokens: int = 8192
    api_key_env: str = "GEMINI_API_KEY"
    top_p: float = 0.95
    top_k: int = 40

    @property
    def api_key(self) -> str:
        """Get API key from environment."""
        key = os.getenv(self.api_key_env)
        if not key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")
        return key


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    sources: List[str] = field(default_factory=lambda: ["usgs", "aqtesolv"])
    validate_metadata: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AppConfig:
    """Main application configuration."""
    retrieval: RetrievalConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    ingestion: IngestionConfig
    logging: LoggingConfig


def load_config(config_name: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file (default, development, production)
                    If None, uses HYDROASSIST_ENV environment variable or 'default'

    Returns:
        AppConfig instance with loaded configuration
    """
    # Determine which config to load
    if config_name is None:
        config_name = os.getenv("HYDROASSIST_ENV", "default")

    # Locate config file
    config_dir = Path(__file__).parent.parent.parent / "configs"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Build config objects
    retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
    vectorstore_config = VectorStoreConfig(**config_dict.get('vectorstore', {}))
    llm_config = LLMConfig(**config_dict.get('llm', {}))
    ingestion_config = IngestionConfig(**config_dict.get('ingestion', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))

    return AppConfig(
        retrieval=retrieval_config,
        vectorstore=vectorstore_config,
        llm=llm_config,
        ingestion=ingestion_config,
        logging=logging_config
    )


def _apply_env_overrides(config_dict: dict) -> dict:
    """
    Apply environment variable overrides to configuration.

    Environment variables follow pattern: HYDROASSIST_SECTION_KEY
    Example: HYDROASSIST_LLM_MODEL
    """
    overrides = {
        'HYDROASSIST_LLM_MODEL': ('llm', 'model'),
        'HYDROASSIST_LLM_TEMPERATURE': ('llm', 'temperature', float),
        'HYDROASSIST_CHUNK_SIZE': ('retrieval', 'chunk_size', int),
        'HYDROASSIST_TOP_K': ('retrieval', 'top_k', int),
        'HYDROASSIST_LOG_LEVEL': ('logging', 'level'),
    }

    for env_var, config_path in overrides.items():
        value = os.getenv(env_var)
        if value is not None:
            section = config_path[0]
            key = config_path[1]
            converter = config_path[2] if len(config_path) > 2 else str

            if section not in config_dict:
                config_dict[section] = {}

            config_dict[section][key] = converter(value)

    return config_dict
