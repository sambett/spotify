"""
Token Manager for storing and retrieving OAuth tokens.
Handles secure token persistence to disk.
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict

from utils import setup_logger


logger = setup_logger(__name__)


class TokenManager:
    """
    Manages OAuth token storage and retrieval.
    
    Stores tokens in JSON file with expiration timestamp.
    Tokens are stored in user's data directory for security.
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize token manager.
        
        Args:
            storage_path: Path to token storage file
        """
        self.storage_path = Path(storage_path)
        
        # Ensure parent directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save_tokens(
        self, 
        access_token: str, 
        refresh_token: Optional[str] = None,
        expires_in: int = 3600
    ) -> None:
        """
        Save tokens to storage.
        
        Args:
            access_token: OAuth access token
            refresh_token: OAuth refresh token (optional)
            expires_in: Token lifetime in seconds
        """
        expiry_time = datetime.now() + timedelta(seconds=expires_in)
        
        token_data = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': expiry_time.isoformat(),
            'saved_at': datetime.now().isoformat()
        }
        
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            logger.debug(f"Tokens saved to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")
            raise
    
    def load_tokens(self) -> Optional[Dict[str, str]]:
        """
        Load tokens from storage.
        
        Returns:
            Token dictionary or None if not found
        """
        if not self.storage_path.exists():
            logger.debug("No stored tokens found")
            return None
        
        try:
            with open(self.storage_path, 'r') as f:
                token_data = json.load(f)
            
            logger.debug("Tokens loaded from storage")
            return token_data
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            return None
    
    def is_token_expired(self) -> bool:
        """
        Check if stored token is expired.
        
        Returns:
            True if expired or not found, False if still valid
        """
        token_data = self.load_tokens()
        
        if not token_data:
            return True
        
        try:
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            # Add 5-minute buffer for safety
            buffer_time = datetime.now() + timedelta(minutes=5)
            
            is_expired = buffer_time >= expires_at
            
            if is_expired:
                logger.debug("Token is expired")
            else:
                time_remaining = (expires_at - datetime.now()).total_seconds() / 60
                logger.debug(f"Token valid for {time_remaining:.1f} more minutes")
            
            return is_expired
        except Exception as e:
            logger.warning(f"Error checking token expiry: {e}")
            return True
    
    def clear_tokens(self) -> None:
        """Delete stored tokens."""
        if self.storage_path.exists():
            self.storage_path.unlink()
            logger.info("Tokens cleared")
