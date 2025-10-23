"""
Main Orchestration Script - Spotify Data Ingestion Pipeline
Fetches data from Spotify API and Kaggle, writes to Bronze layer.
"""
import sys
from pathlib import Path
from datetime import datetime

# PySpark imports
from pyspark.sql import SparkSession

# Local imports
from config import get_config
from clients.auth.spotify_auth import SpotifyAuthClient
from clients.spotify_api import SpotifyAPIClient
from loaders.kaggle_loader import KaggleLoader
from mappers.spotify_mapper import SpotifyMapper
from mappers.kaggle_mapper import KaggleMapper
from writers.delta_writer import DeltaWriter
from utils import setup_logger


logger = setup_logger(__name__)


def create_spark_session(app_name: str) -> SparkSession:
    """Create Spark session with Delta Lake support."""
    logger.info("Creating Spark session...")
    
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"✅ Spark session created: {spark.version}")
    
    return spark


def ingest_listening_history(spark: SparkSession, config, api_client: SpotifyAPIClient) -> None:
    """
    Fetch and ingest listening history from Spotify API.
    
    Steps:
    1. Fetch recently played tracks
    2. Map to Bronze schema
    3. Write to Delta Lake
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Ingesting Listening History")
    logger.info("=" * 60)
    
    # Fetch recently played
    logger.info("Fetching recently played tracks from Spotify API...")
    recently_played = api_client.fetch_all_recently_played(max_items=1000)
    
    if not recently_played:
        logger.warning("No recently played tracks found. Skipping.")
        return
    
    logger.info(f"✅ Fetched {len(recently_played)} listening records")
    
    # Map to schema and create DataFrame
    mapper = SpotifyMapper(spark)
    listening_df = mapper.create_listening_history_df(recently_played)
    
    # Write to Bronze
    DeltaWriter.write_listening_history(
        listening_df,
        config.paths.listening_history,
        mode='append'
    )
    
    logger.info("✅ Listening history ingestion complete\n")


def ingest_tracks_features(spark: SparkSession, config, api_client: SpotifyAPIClient) -> None:
    """
    Fetch and ingest audio features for tracks.
    
    Steps:
    1. Get track IDs from recently played
    2. Fetch audio features (valence, energy, danceability, etc.)
    3. Fetch track metadata
    4. Combine features + metadata
    5. Write to Delta Lake
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Ingesting Track Features")
    logger.info("=" * 60)
    
    # Fetch recently played to get track IDs
    logger.info("Getting track IDs from recently played...")
    recently_played = api_client.fetch_all_recently_played(max_items=1000)
    
    if not recently_played:
        logger.warning("No tracks found. Skipping.")
        return
    
    # Extract unique track IDs
    track_ids = list(set([item['track']['id'] for item in recently_played if 'track' in item and 'id' in item['track']]))
    logger.info(f"Found {len(track_ids)} unique tracks")
    
    # Fetch audio features (CRITICAL for mood analysis!)
    logger.info("Fetching audio features (valence, energy, danceability, etc.)...")
    audio_features = api_client.fetch_audio_features(track_ids)
    
    # Fetch track metadata
    logger.info("Fetching track metadata...")
    tracks_metadata = api_client.fetch_track_details(track_ids)
    
    logger.info(f"✅ Fetched {len(tracks_metadata)} track details and {len(audio_features)} audio features")
    
    # Map to schema and create DataFrame
    mapper = SpotifyMapper(spark)
    tracks_df = mapper.create_tracks_features_df(tracks_metadata, audio_features)
    
    # Write to Bronze (merge mode: update existing, insert new)
    DeltaWriter.write_tracks_features(
        tracks_df,
        config.paths.tracks_features,
        mode='append'
    )
    
    logger.info("✅ Track features ingestion complete\n")


def ingest_kaggle_dataset(spark: SparkSession, config) -> None:
    """
    Load and ingest Kaggle dataset (static catalog of ~100K tracks).
    
    Steps:
    1. Load Kaggle CSV
    2. Map to Bronze schema
    3. Write to Delta Lake (overwrite mode)
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Ingesting Kaggle Dataset")
    logger.info("=" * 60)
    
    if not config.paths.kaggle_csv.exists():
        logger.warning(f"Kaggle CSV not found at {config.paths.kaggle_csv}. Skipping.")
        logger.info("To add Kaggle dataset: Place CSV at data/kaggle/dataset.csv")
        return
    
    # Load Kaggle data
    logger.info(f"Loading Kaggle dataset from {config.paths.kaggle_csv}...")
    
    # Read CSV with Spark
    kaggle_df = spark.read.csv(
        str(config.paths.kaggle_csv),
        header=True,
        inferSchema=True
    )
    
    initial_count = kaggle_df.count()
    logger.info(f"✅ Loaded {initial_count:,} tracks from Kaggle")
    
    # Map to Bronze schema
    mapper = KaggleMapper()
    kaggle_df_mapped = mapper.map_to_bronze_schema(kaggle_df)
    
    # Write to Bronze (overwrite since Kaggle data is static)
    DeltaWriter.write_kaggle_tracks(
        kaggle_df_mapped,
        config.paths.kaggle_tracks,
        mode='overwrite'
    )
    
    logger.info("✅ Kaggle dataset ingestion complete\n")


def run_pipeline():
    """Main pipeline execution."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"🚀 SPOTIFY ANALYTICS PIPELINE - Bronze Ingestion")
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")
    
    try:
        # Load configuration
        config = get_config()
        logger.info("✅ Configuration loaded")
        
        # Create Spark session
        spark = create_spark_session(config.spark_app_name)
        
        # Initialize Spotify API client
        logger.info("Initializing Spotify API client...")
        auth_client = SpotifyAuthClient(config.spotify)
        api_client = SpotifyAPIClient(config, auth_client)
        logger.info("✅ API client initialized\n")
        
        # Run ingestion steps
        ingest_listening_history(spark, config, api_client)
        ingest_tracks_features(spark, config, api_client)
        ingest_kaggle_dataset(spark, config)
        
        # Success summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"⏱️  Duration: {duration:.1f} seconds")
        logger.info(f"📊 Data written to: {config.paths.bronze_base}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Stop Spark
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    sys.exit(run_pipeline())
