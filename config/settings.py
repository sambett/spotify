"""
Centralized configuration from environment variables.
Loads all secrets and settings without hardcoding.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


@dataclass
class SpotifyConfig:
    """Spotify API configuration."""
    client_id: str
    client_secret: str
    redirect_uri: str
    token_storage_path: Path
    scopes: str = "user-read-recently-played"
    
    @classmethod
    def from_env(cls) -> 'SpotifyConfig':
        """Load from environment variables."""
        return cls(
            client_id=os.getenv('CLIENT_ID', ''),
            client_secret=os.getenv('CLIENT_SECRET', ''),
            redirect_uri=os.getenv('REDIRECT_URI', 'http://127.0.0.1:8888/callback'),
            token_storage_path=Path(os.getenv('TOKEN_PATH', 'data/.spotify_tokens.json')),
            scopes=os.getenv('SPOTIFY_SCOPES', 'user-read-recently-played')
        )
    
    def validate(self) -> None:
        """Validate required fields are present."""
        if not self.client_id:
            raise ValueError("CLIENT_ID environment variable is required")
        if not self.client_secret:
            raise ValueError("CLIENT_SECRET environment variable is required")


@dataclass
class PathConfig:
    """Data paths configuration."""
    bronze_base: Path
    kaggle_csv: Path
    
    listening_history: Path
    tracks_features: Path
    kaggle_tracks: Path
    
    @classmethod
    def from_env(cls) -> 'PathConfig':
        """Load from environment with defaults."""
        bronze_base = Path(os.getenv('BRONZE_PATH', 'data/bronze'))
        
        return cls(
            bronze_base=bronze_base,
            kaggle_csv=Path(os.getenv('KAGGLE_CSV', 'data/kaggle/dataset.csv')),
            listening_history=bronze_base / 'listening_history_bronze',
            tracks_features=bronze_base / 'my_tracks_features_bronze',
            kaggle_tracks=bronze_base / 'kaggle_tracks_bronze'
        )


@dataclass
class AppConfig:
    """Application-wide configuration."""
    spotify: SpotifyConfig
    paths: PathConfig
    spark_app_name: str = "SpotifyBronzeIngestion"
    log_level: str = "INFO"
    batch_size: int = 50  # Spotify API max per request
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def load(cls, env_file: Optional[str] = None) -> 'AppConfig':
        """
        Load all configuration.
        
        Args:
            env_file: Path to .env file (optional, will search parent dirs)
        """
        # Load environment from .env file if it exists
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Searches parent directories
        
        config = cls(
            spotify=SpotifyConfig.from_env(),
            paths=PathConfig.from_env(),
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )
        
        # Validate critical settings
        config.spotify.validate()
        
        return config


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create configuration singleton."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None
