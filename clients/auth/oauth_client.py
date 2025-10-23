"""
OAuth Client wrapper.
Provides a simple interface for OAuth operations.
"""
from typing import Optional

from config import SpotifyConfig
from .spotify_auth import SpotifyAuthClient
from utils import setup_logger


logger = setup_logger(__name__)


class OAuthClient:
    """
    Simple OAuth client wrapper.
    Provides convenience methods for common OAuth operations.
    """
    
    def __init__(self, config: SpotifyConfig):
        """
        Initialize OAuth client.
        
        Args:
            config: Spotify configuration
        """
        self.config = config
        self.auth_client = SpotifyAuthClient(config)
    
    def authenticate(self) -> None:
        """Perform OAuth authentication flow."""
        self.auth_client.authenticate()
    
    def get_token(self) -> str:
        """
        Get a valid access token.
        
        Returns:
            Access token
        """
        return self.auth_client.get_valid_token()
    
    def is_authenticated(self) -> bool:
        """
        Check if we have a valid token.
        
        Returns:
            True if authenticated, False otherwise
        """
        try:
            tokens = self.auth_client.token_manager.load_tokens()
            if not tokens:
                return False
            
            return not self.auth_client.token_manager.is_token_expired()
        except Exception:
            return False
    
    def clear_tokens(self) -> None:
        """Clear stored tokens."""
        self.auth_client.token_manager.clear_tokens()
        logger.info("Tokens cleared - re-authentication required")


__all__ = ['OAuthClient']
