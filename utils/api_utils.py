"""
API Utilities Module

Single Responsibility: Provide retry logic and error handling
- Retry failed requests with exponential backoff
- Handle common HTTP errors
- Parse and validate API responses

This module contains helper functions for robust API communication.
"""

import time
from typing import Callable, Any, Optional
import requests


class APIError(Exception):
    """Raised when API requests fail after retries."""
    pass


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: tuple = (requests.exceptions.RequestException,),
    status_codes_to_retry: tuple = (429, 500, 502, 503, 504)
) -> Callable:
    """
    Decorator: Retry function with exponential backoff on failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        exceptions: Exceptions to catch and retry on
        status_codes_to_retry: HTTP codes to retry on
    
    Usage:
        @retry_on_failure()
        def api_call():
            response = requests.get(...)
            return validate_response(response)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # If result is a response, check for retryable status
                    if isinstance(result, requests.Response) and result.status_code in status_codes_to_retry:
                        raise APIError(f"Retryable status: {result.status_code}")
                    return result
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Check if retryable
                        if hasattr(e, 'response') and e.response.status_code in status_codes_to_retry:
                            print(f"⚠️  Error {e.response.status_code}, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                        else:
                            print(f"⚠️  Network error, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise APIError(f"Max retries exceeded: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


def validate_response(response: requests.Response) -> dict:
    """
    Validate and parse API response.
    
    Args:
        response: requests.Response object
        
    Returns:
        Parsed JSON response
        
    Raises:
        APIError: If response is invalid (non-2xx or bad JSON)
    """
    # Check status code
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Try to get error message from response (Spotify JSON format)
        try:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', str(e))
        except:
            error_msg = str(e)
        raise APIError(f"API request failed: {error_msg}")
    
    # Parse JSON
    try:
        data = response.json()
    except ValueError as e:
        raise APIError(f"Invalid JSON response: {e}")
    
    return data


def extract_track_info(track_data: dict, played_at: Optional[str] = None) -> dict:
    """
    Extract relevant track information from Spotify API response.
    
    Args:
        track_data: Track object from Spotify API
        played_at: Optional timestamp when track was played (ISO-8601)
        
    Returns:
        Dict with standardized track information
    """
    # Safely extract nested data
    track = track_data.get('track', track_data)
    
    # Extract basic info
    track_info = {
        'track_id': track.get('id'),
        'track_name': track.get('name'),
        'duration_ms': track.get('duration_ms'),
        'explicit': track.get('explicit', False),
        'popularity': track.get('popularity'),
    }
    
    # Extract artists
    artists = track.get('artists', [])
    track_info['artists'] = [artist.get('name') for artist in artists]
    track_info['artist_ids'] = [artist.get('id') for artist in artists]
    
    # Extract album info
    album = track.get('album', {})
    track_info['album_name'] = album.get('name')
    track_info['album_id'] = album.get('id')
    track_info['release_date'] = album.get('release_date')
    
    # Add played_at if provided
    if played_at:
        track_info['played_at'] = played_at
    
    return track_info