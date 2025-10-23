"""
Authentication module for Spotify OAuth.
Handles OAuth flow, token management, and persistence.
"""
from .spotify_auth import SpotifyAuthClient
from .token_manager import TokenManager
from .oauth_client import OAuthClient

__all__ = [
    'SpotifyAuthClient',
    'TokenManager',
    'OAuthClient'
]
