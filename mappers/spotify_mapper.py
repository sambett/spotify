"""
Spotify API data mapper.
Transforms raw API responses into Bronze schema format.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import Row

from schemas import get_listening_history_schema, get_tracks_features_schema
from utils import setup_logger


logger = setup_logger(__name__)


class SpotifyMapper:
    """
    Maps Spotify API responses to Bronze schemas.
    
    Responsibilities:
    - Extract relevant fields from API JSON
    - Handle missing or malformed data
    - Type conversions
    - Add ingestion timestamp
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize mapper.
        
        Args:
            spark: Spark session
        """
        self.spark = spark
    
    @staticmethod
    def map_listening_history_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Map a single recently-played item to listening_history schema.
        
        Args:
            item: Raw API item from recently-played endpoint
        
        Returns:
            Mapped dictionary or None if invalid
        """
        try:
            track = item.get('track', {})
            
            # Extract required fields
            track_id = track.get('id')
            played_at = item.get('played_at')
            
            if not track_id or not played_at:
                logger.warning(f"Missing required fields: track_id={track_id}, played_at={played_at}")
                return None
            
            # Parse played_at timestamp
            try:
                played_at_dt = datetime.fromisoformat(played_at.replace('Z', '+00:00'))
            except ValueError as e:
                logger.warning(f"Invalid played_at timestamp: {played_at}")
                return None
            
            # Extract artist name (first artist)
            artists = track.get('artists', [])
            artist_name = artists[0].get('name') if artists else None
            
            # Extract album name
            album = track.get('album', {})
            album_name = album.get('name')
            
            return {
                'track_id': track_id,
                'played_at': played_at_dt,
                'track_name': track.get('name'),
                'artist_name': artist_name,
                'album_name': album_name,
                'duration_ms': track.get('duration_ms'),
                '_ingested_at': datetime.now()
            }
        
        except Exception as e:
            logger.warning(f"Failed to map listening history item: {e}")
            return None
    
    @staticmethod
    def map_track_with_features(
        track: Dict[str, Any], 
        audio_features: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Map track details + audio features to tracks_features schema.
        
        CRITICAL: This combines track metadata with audio features (danceability,
        energy, valence, etc.) that are essential for mood analysis.
        
        Args:
            track: Track details from /tracks endpoint
            audio_features: Audio features from /audio-features endpoint
        
        Returns:
            Mapped dictionary or None if invalid
        """
        try:
            track_id = track.get('id')
            
            if not track_id:
                logger.warning("Missing track_id in track data")
                return None
            
            # Extract artist name
            artists = track.get('artists', [])
            artist_name = artists[0].get('name') if artists else None
            
            # Extract album name
            album = track.get('album', {})
            album_name = album.get('name')
            
            # Combine track metadata with audio features
            return {
                'track_id': track_id,
                'track_name': track.get('name'),
                'artist_name': artist_name,
                'album_name': album_name,
                'popularity': track.get('popularity'),
                'duration_ms': track.get('duration_ms'),
                'explicit': track.get('explicit'),
                # Audio features - the core of mood analysis
                'danceability': audio_features.get('danceability'),
                'energy': audio_features.get('energy'),
                'acousticness': audio_features.get('acousticness'),
                'instrumentalness': audio_features.get('instrumentalness'),
                'valence': audio_features.get('valence'),
                'tempo': audio_features.get('tempo'),
                'time_signature': audio_features.get('time_signature'),
                '_ingested_at': datetime.now()
            }
        
        except Exception as e:
            logger.warning(f"Failed to map track with features: {e}")
            return None
    
    def create_listening_history_df(self, items: List[Dict[str, Any]]) -> DataFrame:
        """
        Create DataFrame from recently-played items.
        
        Args:
            items: List of raw API items
        
        Returns:
            DataFrame with listening_history_bronze schema
        """
        logger.info(f"Mapping {len(items)} listening history items...")
        
        # Map items
        mapped_items = []
        for item in items:
            mapped = self.map_listening_history_item(item)
            if mapped:
                mapped_items.append(mapped)
        
        logger.info(f"✅ Successfully mapped {len(mapped_items)}/{len(items)} items")
        
        if not mapped_items:
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], get_listening_history_schema())
        
        # Create DataFrame
        df = self.spark.createDataFrame(
            [Row(**item) for item in mapped_items],
            get_listening_history_schema()
        )
        
        return df
    
    def create_tracks_features_df(
        self, 
        tracks: List[Dict[str, Any]], 
        audio_features_list: List[Dict[str, Any]]
    ) -> DataFrame:
        """
        Create DataFrame from track details and audio features.
        
        Args:
            tracks: List of track detail dictionaries
            audio_features_list: List of audio feature dictionaries
        
        Returns:
            DataFrame with my_tracks_features_bronze schema
        """
        logger.info(f"Mapping {len(tracks)} tracks with audio features...")
        
        # Build lookup dict for audio features by track_id
        features_by_id = {
            f['id']: f 
            for f in audio_features_list 
            if f and 'id' in f
        }
        
        logger.debug(f"Audio features available for {len(features_by_id)} tracks")
        
        # Map tracks with their features
        mapped_items = []
        for track in tracks:
            track_id = track.get('id')
            if not track_id:
                continue
            
            # Get audio features for this track
            audio_features = features_by_id.get(track_id, {})
            
            if not audio_features:
                logger.warning(f"No audio features for track {track_id}")
                # Still include the track, audio features will be NULL
            
            mapped = self.map_track_with_features(track, audio_features)
            if mapped:
                mapped_items.append(mapped)
        
        logger.info(f"✅ Successfully mapped {len(mapped_items)}/{len(tracks)} tracks")
        
        if not mapped_items:
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], get_tracks_features_schema())
        
        # Create DataFrame
        df = self.spark.createDataFrame(
            [Row(**item) for item in mapped_items],
            get_tracks_features_schema()
        )
        
        return df
