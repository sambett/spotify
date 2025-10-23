"""
OAuth Callback Server.
Simple HTTP server to catch OAuth redirect callback.
"""
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from utils import setup_logger


logger = setup_logger(__name__)


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""
    
    authorization_code: Optional[str] = None
    
    def do_GET(self):
        """Handle GET request (OAuth callback)."""
        # Parse query parameters
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        
        # Extract authorization code
        if 'code' in query_params:
            OAuthCallbackHandler.authorization_code = query_params['code'][0]
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            success_html = """
            <html>
            <head><title>Spotify Authentication</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: #1DB954;">✅ Authentication Successful!</h1>
                <p>You can close this window and return to your application.</p>
            </body>
            </html>
            """
            self.wfile.write(success_html.encode())
            
            logger.info("✅ Authorization code received")
        else:
            # Error in callback
            error = query_params.get('error', ['Unknown error'])[0]
            
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            error_html = f"""
            <html>
            <head><title>Spotify Authentication</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: #E74C3C;">❌ Authentication Failed</h1>
                <p>Error: {error}</p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())
            
            logger.error(f"OAuth error: {error}")
    
    def log_message(self, format, *args):
        """Suppress default server logging."""
        pass


class OAuthCallbackServer:
    """
    Temporary HTTP server to catch OAuth redirect.
    Starts server, waits for callback, then shuts down.
    """
    
    def __init__(self, redirect_uri: str):
        """
        Initialize callback server.
        
        Args:
            redirect_uri: Expected redirect URI (e.g., http://127.0.0.1:8888/callback)
        """
        self.redirect_uri = redirect_uri
        
        # Parse host and port from redirect URI
        parsed = urllib.parse.urlparse(redirect_uri)
        self.host = parsed.hostname or '127.0.0.1'
        self.port = parsed.port or 8888
    
    def wait_for_callback(self, timeout: int = 300) -> Optional[str]:
        """
        Start server and wait for OAuth callback.
        
        Args:
            timeout: Maximum wait time in seconds (default 5 minutes)
        
        Returns:
            Authorization code or None if timeout
        """
        logger.info(f"Starting callback server on {self.host}:{self.port}")
        logger.info("Waiting for authorization...")
        
        # Reset authorization code
        OAuthCallbackHandler.authorization_code = None
        
        # Create server
        server = HTTPServer((self.host, self.port), OAuthCallbackHandler)
        server.timeout = timeout
        
        try:
            # Handle one request (the callback)
            server.handle_request()
            
            return OAuthCallbackHandler.authorization_code
        except Exception as e:
            logger.error(f"Callback server error: {e}")
            return None
        finally:
            server.server_close()
