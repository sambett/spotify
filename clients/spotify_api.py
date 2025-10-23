"""
Spotify Web API client.
Fetches listening history and audio features.
"""
import requests
import time
from typing import List, Dict, Any, Optional

from config import AppConfig
from clients.spotify_auth import SpotifyAuthClient
from utils import setup_logger


logger = setup_logger(__name__)


class SpotifyAPIClient:
    """
    Spotify Web API client for fetching data.
    
    Responsibilities:
    - Fetch recently played tracks
    - Fetch audio features for tracks
    - Handle pagination
    - Implement retry logic with exponential backoff
    - Respect rate limits
    """
    
    BASE_URL = "https://api.spotify.com/v1"
    
    def __init__(self, config: AppConfig, auth_client: SpotifyAuthClient):
        """
        Initialize API client.
        
        Args:
            config: Application configuration
            auth_client: Authentication client
        """
        self.config = config
        self.auth_client = auth_client
        self.session = requests.Session()
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        method: str = 'GET'
    ) -> Dict:
        """
        Make authenticated API request with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., '/me/player/recently-played')
            params: Query parameters
            method: HTTP method
        
        Returns:
            Response JSON
        
        Raises:
            RuntimeError: If all retries fail
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                # Get valid token
                token = self.auth_client.get_valid_token()
                
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }
                
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    timeout=30
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                # Handle other errors
                if response.status_code >= 400:
                    logger.warning(f"API error {response.status_code}: {response.text}")
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.debug(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"API request failed: {response.text}")
                
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API request failed after {self.config.max_retries} attempts: {e}")
        
        raise RuntimeError("Max retries exceeded")
    
    def fetch_recently_played(
        self, 
        limit: int = 50,
        after: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recently played tracks.
        
        Args:
            limit: Number of tracks to fetch (max 50)
            after: Unix timestamp in milliseconds (fetch tracks after this time)
        
        Returns:
            List of recently played track items
        """
        params = {'limit': min(limit, 50)}
        if after:
            params['after'] = after
        
        logger.debug(f"Fetching recently played (limit={limit}, after={after})")
        
        data = self._make_request('/me/player/recently-played', params=params)
        items = data.get('items', [])
        
        logger.debug(f"Fetched {len(items)} recently played tracks")
        return items
    
    def fetch_audio_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch audio features for multiple tracks.
        
        CRITICAL: This fetches the audio features (danceability, energy, valence, etc.)
        that are essential for mood-based analysis. This was missing in the previous
        implementation.
        
        Args:
            track_ids: List of Spotify track IDs (max 100 per request)
        
        Returns:
            List of audio feature dictionaries
        """
        if not track_ids:
            return []
        
        # Spotify allows max 100 IDs per request
        if len(track_ids) > 100:
            logger.warning(f"Splitting {len(track_ids)} tracks into batches of 100")
            all_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                features = self.fetch_audio_features(batch)
                all_features.extend(features)
            return all_features
        
        params = {'ids': ','.join(track_ids)}
        
        logger.debug(f"Fetching audio features for {len(track_ids)} tracks")
        
        data = self._make_request('/audio-features', params=params)
        features = data.get('audio_features', [])
        
        # Filter out None values (tracks without features)
        features = [f for f in features if f is not None]
        
        logger.debug(f"Fetched audio features for {len(features)} tracks")
        return features
    
    def fetch_track_details(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch full track details including metadata.
        
        Args:
            track_ids: List of Spotify track IDs (max 50 per request)
        
        Returns:
            List of track detail dictionaries
        """
        if not track_ids:
            return []
        
        # Spotify allows max 50 IDs per request for tracks endpoint
        if len(track_ids) > 50:
            logger.warning(f"Splitting {len(track_ids)} tracks into batches of 50")
            all_tracks = []
            for i in range(0, len(track_ids), 50):
                batch = track_ids[i:i+50]
                tracks = self.fetch_track_details(batch)
                all_tracks.extend(tracks)
            return all_tracks
        
        params = {'ids': ','.join(track_ids)}
        
        logger.debug(f"Fetching track details for {len(track_ids)} tracks")
        
        data = self._make_request('/tracks', params=params)
        tracks = data.get('tracks', [])
        
        # Filter out None values
        tracks = [t for t in tracks if t is not None]
        
        logger.debug(f"Fetched details for {len(tracks)} tracks")
        return tracks
    
    def fetch_all_recently_played(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch all available recently played tracks with pagination.
        
        Args:
            max_items: Maximum items to fetch (None = fetch all available)
        
        Returns:
            List of all recently played items
        """
        all_items = []
        after = None
        page = 1
        
        logger.info("Fetching all recently played tracks...")
        
        while True:
            items = self.fetch_recently_played(limit=50, after=after)
            
            if not items:
                break
            
            all_items.extend(items)
            logger.debug(f"Page {page}: {len(items)} tracks (total: {len(all_items)})")
            
            # Check if we've reached max_items
            if max_items and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break
            
            # Get cursor for next page
            # Spotify returns items in reverse chronological order
            # Use the timestamp of the last item as 'after' cursor
            try:
                last_played_at = items[-1]['played_at']
                # Convert ISO timestamp to Unix milliseconds
                from datetime import datetime
                dt = datetime.fromisoformat(last_played_at.replace('Z', '+00:00'))
                after = int(dt.timestamp() * 1000)
            except (KeyError, IndexError, ValueError) as e:
                logger.warning(f"Couldn't extract pagination cursor: {e}")
                break
            
            page += 1
            
            # Add small delay to be nice to the API
            time.sleep(0.1)
        
        logger.info(f"✅ Fetched {len(all_items)} total recently played tracks")
        return all_items
