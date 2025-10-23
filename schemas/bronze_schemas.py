"""
Bronze layer schema definitions.
Exact field specifications for all three Bronze tables.
"""
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    IntegerType, FloatType, BooleanType, TimestampType
)


def get_listening_history_schema() -> StructType:
    """
    Schema for listening_history_bronze.
    Personal Spotify listening events with timestamps.
    """
    return StructType([
        StructField("track_id", StringType(), nullable=False),
        StructField("played_at", TimestampType(), nullable=False),
        StructField("track_name", StringType(), nullable=True),
        StructField("artist_name", StringType(), nullable=True),
        StructField("album_name", StringType(), nullable=True),
        StructField("duration_ms", LongType(), nullable=True),
        StructField("_ingested_at", TimestampType(), nullable=False),
    ])


def get_tracks_features_schema() -> StructType:
    """
    Schema for my_tracks_features_bronze.
    Audio features for tracks in listening history.
    
    CRITICAL: This schema includes audio features that were missing
    in the previous implementation. Features like danceability, energy,
    valence, etc. are essential for mood-based analysis.
    """
    return StructType([
        StructField("track_id", StringType(), nullable=False),
        StructField("track_name", StringType(), nullable=True),
        StructField("artist_name", StringType(), nullable=True),
        StructField("album_name", StringType(), nullable=True),
        StructField("popularity", IntegerType(), nullable=True),
        StructField("duration_ms", LongType(), nullable=True),
        StructField("explicit", BooleanType(), nullable=True),
        # Audio features - the core of mood analysis
        StructField("danceability", FloatType(), nullable=True),
        StructField("energy", FloatType(), nullable=True),
        StructField("acousticness", FloatType(), nullable=True),
        StructField("instrumentalness", FloatType(), nullable=True),
        StructField("valence", FloatType(), nullable=True),
        StructField("tempo", FloatType(), nullable=True),
        StructField("time_signature", IntegerType(), nullable=True),
        StructField("_ingested_at", TimestampType(), nullable=False),
    ])


def get_kaggle_tracks_schema() -> StructType:
    """
    Schema for kaggle_tracks_bronze.
    Static catalog from Kaggle dataset with audio features.
    """
    return StructType([
        StructField("track_id", StringType(), nullable=False),
        StructField("artists", StringType(), nullable=True),
        StructField("album_name", StringType(), nullable=True),
        StructField("track_name", StringType(), nullable=True),
        StructField("popularity", IntegerType(), nullable=True),
        StructField("duration_ms", LongType(), nullable=True),
        StructField("explicit", BooleanType(), nullable=True),
        # Audio features
        StructField("danceability", FloatType(), nullable=True),
        StructField("energy", FloatType(), nullable=True),
        StructField("valence", FloatType(), nullable=True),
        StructField("acousticness", FloatType(), nullable=True),
        StructField("instrumentalness", FloatType(), nullable=True),
        StructField("tempo", FloatType(), nullable=True),
        StructField("track_genre", StringType(), nullable=True),
        StructField("_ingested_at", TimestampType(), nullable=False),
    ])


def get_schema_by_name(name: str) -> StructType:
    """
    Get schema by table name.
    
    Args:
        name: Table name (listening_history_bronze, my_tracks_features_bronze, kaggle_tracks_bronze)
    
    Returns:
        Corresponding schema
    """
    schemas = {
        'listening_history_bronze': get_listening_history_schema(),
        'my_tracks_features_bronze': get_tracks_features_schema(),
        'kaggle_tracks_bronze': get_kaggle_tracks_schema()
    }
    
    if name not in schemas:
        raise ValueError(f"Unknown schema name: {name}")
    
    return schemas[name]
