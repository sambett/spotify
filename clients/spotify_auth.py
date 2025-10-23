"""
Spotify OAuth 2.0 authentication client.
Handles token acquisition, refresh, and persistence.
"""
import json
import requests
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
import threading

from config import SpotifyConfig
from utils import setup_logger


logger = setup_logger(__name__)


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""
    
    auth_code: Optional[str] = None
    error: Optional[str] = None
    
    def do_GET(self):
        """Handle GET request from Spotify redirect."""
        # Parse query parameters
        query_components = parse_qs(urlparse(self.path).query)
        
        if 'code' in query_components:
            CallbackHandler.auth_code = query_components['code'][0]
            message = "✅ Authorization successful! You can close this window."
        elif 'error' in query_components:
            CallbackHandler.error = query_components['error'][0]
            message = f"❌ Authorization failed: {CallbackHandler.error}"
        else:
            message = "⚠️ Unexpected callback"
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(f"""
        <html>
        <head><title>Spotify Auth</title></head>
        <body>
            <h1>{message}</h1>
            <p>This window can be closed.</p>
        </body>
        </html>
        """.encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class SpotifyAuthClient:
    """
    Spotify OAuth 2.0 authentication client.
    
    Responsibilities:
    - Generate authorization URL
    - Handle OAuth callback
    - Exchange authorization code for tokens
    - Refresh access tokens
    - Persist tokens to disk
    """
    
    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(self, config: SpotifyConfig):
        """
        Initialize auth client.
        
        Args:
            config: Spotify configuration
        """
        self.config = config
        self.tokens: Optional[Dict] = None
        self._load_tokens()
    
    def _load_tokens(self) -> None:
        """Load tokens from storage if they exist."""
        if self.config.token_storage_path.exists():
            try:
                with open(self.config.token_storage_path, 'r') as f:
                    self.tokens = json.load(f)
                logger.debug("Loaded existing tokens from storage")
            except Exception as e:
                logger.warning(f"Failed to load tokens: {e}")
                self.tokens = None
    
    def _save_tokens(self, tokens: Dict) -> None:
        """
        Save tokens to storage.
        
        Args:
            tokens: Token dictionary with access_token, refresh_token, expires_at
        """
        self.tokens = tokens
        
        # Ensure directory exists
        self.config.token_storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(self.config.token_storage_path, 'w') as f:
            json.dump(tokens, f, indent=2)
        
        logger.debug(f"Saved tokens to {self.config.token_storage_path}")
    
    def get_authorization_url(self) -> str:
        """
        Generate Spotify authorization URL.
        
        Returns:
            Authorization URL for user to visit
        """
        params = {
            'client_id': self.config.client_id,
            'response_type': 'code',
            'redirect_uri': self.config.redirect_uri,
            'scope': self.config.scopes
        }
        
        return f"{self.AUTH_URL}?{urlencode(params)}"
    
    def wait_for_callback(self, timeout: int = 300) -> tuple[Optional[str], Optional[str]]:
        """
        Start local server and wait for OAuth callback.
        
        Args:
            timeout: Maximum seconds to wait
        
        Returns:
            Tuple of (auth_code, error)
        """
        # Parse redirect URI for host/port
        parsed = urlparse(self.config.redirect_uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 8888
        
        # Reset class variables
        CallbackHandler.auth_code = None
        CallbackHandler.error = None
        
        # Start server
        server = HTTPServer((host, port), CallbackHandler)
        
        # Run server in thread with timeout
        server_thread = threading.Thread(target=server.handle_request)
        server_thread.daemon = True
        server_thread.start()
        server_thread.join(timeout=timeout)
        
        # Cleanup
        try:
            server.server_close()
        except:
            pass
        
        return CallbackHandler.auth_code, CallbackHandler.error
    
    def exchange_code_for_tokens(self, auth_code: str) -> Dict:
        """
        Exchange authorization code for access/refresh tokens.
        
        Args:
            auth_code: Authorization code from callback
        
        Returns:
            Token dictionary
        
        Raises:
            RuntimeError: If token exchange fails
        """
        data = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': self.config.redirect_uri,
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        
        if response.status_code != 200:
            raise RuntimeError(f"Token exchange failed: {response.text}")
        
        token_data = response.json()
        
        # Calculate expiration time
        expires_in = token_data.get('expires_in', 3600)
        expires_at = (datetime.now() + timedelta(seconds=expires_in)).isoformat()
        
        tokens = {
            'access_token': token_data['access_token'],
            'refresh_token': token_data.get('refresh_token'),
            'expires_at': expires_at
        }
        
        self._save_tokens(tokens)
        return tokens
    
    def refresh_access_token(self) -> Dict:
        """
        Refresh access token using refresh token.
        
        Returns:
            New token dictionary
        
        Raises:
            RuntimeError: If refresh fails or no refresh token available
        """
        if not self.tokens or not self.tokens.get('refresh_token'):
            raise RuntimeError("No refresh token available. Re-authorization required.")
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.tokens['refresh_token'],
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        
        if response.status_code != 200:
            raise RuntimeError(f"Token refresh failed: {response.text}")
        
        token_data = response.json()
        
        # Calculate expiration
        expires_in = token_data.get('expires_in', 3600)
        expires_at = (datetime.now() + timedelta(seconds=expires_in)).isoformat()
        
        # Update tokens (keep refresh_token if not provided)
        tokens = {
            'access_token': token_data['access_token'],
            'refresh_token': token_data.get('refresh_token', self.tokens['refresh_token']),
            'expires_at': expires_at
        }
        
        self._save_tokens(tokens)
        logger.debug("Refreshed access token")
        return tokens
    
    def get_valid_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token
        
        Raises:
            RuntimeError: If no tokens available or refresh fails
        """
        if not self.tokens:
            raise RuntimeError("Not authorized. Run authorization flow first.")
        
        # Check if token is expired
        expires_at = datetime.fromisoformat(self.tokens['expires_at'])
        
        if datetime.now() >= expires_at - timedelta(minutes=5):
            # Token expired or expiring soon, refresh it
            logger.debug("Token expired, refreshing...")
            self.refresh_access_token()
        
        return self.tokens['access_token']
    
    def authorize(self) -> None:
        """
        Run complete authorization flow.
        
        Steps:
        1. Open authorization URL in browser
        2. Wait for callback
        3. Exchange code for tokens
        4. Save tokens
        """
        logger.info("🔐 Starting Spotify authorization...")
        
        # Generate auth URL
        auth_url = self.get_authorization_url()
        logger.info(f"Opening browser for authorization...")
        logger.info(f"If browser doesn't open, visit: {auth_url}")
        
        # Open browser
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            logger.warning(f"Couldn't open browser: {e}")
        
        # Wait for callback
        logger.info("Waiting for authorization callback...")
        auth_code, error = self.wait_for_callback()
        
        if error:
            raise RuntimeError(f"Authorization failed: {error}")
        
        if not auth_code:
            raise RuntimeError("No authorization code received")
        
        logger.info("✅ Authorization code received")
        
        # Exchange for tokens
        logger.info("Exchanging code for tokens...")
        self.exchange_code_for_tokens(auth_code)
        
        logger.info("✅ Authorization complete!")
    
    def is_authorized(self) -> bool:
        """Check if we have valid tokens."""
        return self.tokens is not None and 'access_token' in self.tokens
