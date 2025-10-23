"""
Spotify OAuth Authentication Client.
Handles OAuth 2.0 flow with PKCE and token management.
"""
import webbrowser
import urllib.parse
import secrets
import hashlib
import base64
from pathlib import Path
from typing import Optional, Dict

from config import SpotifyConfig
from .token_manager import TokenManager
from .oauth_server import OAuthCallbackServer
from utils import setup_logger


logger = setup_logger(__name__)


class SpotifyAuthClient:
    """
    Manages Spotify OAuth authentication.
    
    Responsibilities:
    - Initiate OAuth flow
    - Handle callback
    - Manage access/refresh tokens
    - Automatically refresh expired tokens
    """
    
    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(self, config: SpotifyConfig):
        """
        Initialize auth client.
        
        Args:
            config: Spotify API configuration
        """
        self.config = config
        self.token_manager = TokenManager(config.token_storage_path)
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier."""
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    def get_authorization_url(self, code_challenge: str) -> str:
        """
        Build authorization URL for OAuth flow.
        
        Args:
            code_challenge: PKCE code challenge
        
        Returns:
            Authorization URL
        """
        params = {
            'client_id': self.config.client_id,
            'response_type': 'code',
            'redirect_uri': self.config.redirect_uri,
            'scope': self.config.scopes,
            'code_challenge_method': 'S256',
            'code_challenge': code_challenge
        }
        
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"
    
    def authenticate(self) -> None:
        """
        Perform OAuth authentication flow.
        Opens browser for user authorization.
        """
        logger.info("Starting OAuth authentication...")
        
        # Generate PKCE parameters
        code_verifier = self._generate_code_verifier()
        code_challenge = self._generate_code_challenge(code_verifier)
        
        # Build auth URL
        auth_url = self.get_authorization_url(code_challenge)
        
        logger.info("Opening browser for authorization...")
        logger.info(f"If browser doesn't open, visit: {auth_url}")
        
        # Open browser
        webbrowser.open(auth_url)
        
        # Start callback server to catch redirect
        server = OAuthCallbackServer(self.config.redirect_uri)
        authorization_code = server.wait_for_callback()
        
        if not authorization_code:
            raise RuntimeError("Failed to get authorization code")
        
        logger.info("✅ Authorization code received")
        
        # Exchange code for tokens
        self._exchange_code_for_tokens(authorization_code, code_verifier)
    
    def _exchange_code_for_tokens(self, code: str, code_verifier: str) -> None:
        """
        Exchange authorization code for access/refresh tokens.
        
        Args:
            code: Authorization code
            code_verifier: PKCE code verifier
        """
        import requests
        
        logger.info("Exchanging authorization code for tokens...")
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.config.redirect_uri,
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code_verifier': code_verifier
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        
        if response.status_code != 200:
            raise RuntimeError(f"Token exchange failed: {response.text}")
        
        tokens = response.json()
        
        self._access_token = tokens['access_token']
        self._refresh_token = tokens.get('refresh_token')
        expires_in = tokens.get('expires_in', 3600)
        
        # Save tokens
        self.token_manager.save_tokens(
            access_token=self._access_token,
            refresh_token=self._refresh_token,
            expires_in=expires_in
        )
        
        logger.info("✅ Tokens obtained and saved")
    
    def _refresh_access_token(self) -> None:
        """Refresh the access token using refresh token."""
        import requests
        
        if not self._refresh_token:
            logger.warning("No refresh token available, need to re-authenticate")
            self.authenticate()
            return
        
        logger.info("Refreshing access token...")
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self._refresh_token,
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        
        if response.status_code != 200:
            logger.warning("Token refresh failed, re-authenticating...")
            self.authenticate()
            return
        
        tokens = response.json()
        
        self._access_token = tokens['access_token']
        # Refresh token might be updated
        if 'refresh_token' in tokens:
            self._refresh_token = tokens['refresh_token']
        expires_in = tokens.get('expires_in', 3600)
        
        # Save updated tokens
        self.token_manager.save_tokens(
            access_token=self._access_token,
            refresh_token=self._refresh_token,
            expires_in=expires_in
        )
        
        logger.info("✅ Access token refreshed")
    
    def get_valid_token(self) -> str:
        """
        Get a valid access token.
        Loads from storage, refreshes if expired, or re-authenticates if needed.
        
        Returns:
            Valid access token
        """
        # Try to load from storage
        if not self._access_token:
            stored_tokens = self.token_manager.load_tokens()
            
            if stored_tokens:
                self._access_token = stored_tokens['access_token']
                self._refresh_token = stored_tokens.get('refresh_token')
                
                # Check if expired
                if self.token_manager.is_token_expired():
                    logger.info("Token expired, refreshing...")
                    self._refresh_access_token()
            else:
                # No stored tokens, need to authenticate
                logger.info("No stored tokens found, authenticating...")
                self.authenticate()
        
        # If still no token, something went wrong
        if not self._access_token:
            raise RuntimeError("Failed to obtain valid access token")
        
        return self._access_token
