"""
Spotify API data ingestion with optimized Kaggle handling.
"""
import sys
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession
from config import get_config
from clients.auth.spotify_auth import SpotifyAuthClient
from clients.spotify_api import SpotifyAPIClient
from mappers.spotify_mapper import SpotifyMapper
from mappers.kaggle_mapper import KaggleMapper
from writers.delta_writer import DeltaWriter
from utils import setup_logger


logger = setup_logger(__name__)


def create_spark_session(app_name: str) -> SparkSession:
    """Create Spark session with optimized settings for large datasets."""
    logger.info("Creating Spark session...")
    
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "10")
        .config("spark.default.parallelism", "10")
        # Enable better error tracking
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"✅ Spark session created: {spark.version}")
    return spark


def ingest_listening_history(spark: SparkSession, config, api_client: SpotifyAPIClient) -> None:
    """Fetch and ingest listening history from Spotify API."""
    logger.info("=" * 60)
    logger.info("STEP 1: Ingesting Listening History")
    logger.info("=" * 60)

    try:
        recently_played = api_client.fetch_all_recently_played(max_items=1000)

        if not recently_played:
            logger.warning("No recently played tracks found.")
            return

        logger.info(f"✅ Fetched {len(recently_played)} listening records")

        mapper = SpotifyMapper(spark)
        listening_df = mapper.create_listening_history_df(recently_played)

        DeltaWriter.write_listening_history(
            listening_df,
            config.paths.listening_history,
            mode='append'
        )

        logger.info("✅ Listening history ingestion complete\n")

    except Exception as e:
        logger.error(f"⚠️  Listening history ingestion failed: {e}")
        logger.info("Continuing with remaining ingestion steps...\n")


def ingest_tracks_features(spark: SparkSession, config, api_client: SpotifyAPIClient) -> None:
    """Fetch and ingest audio features for tracks."""
    logger.info("=" * 60)
    logger.info("STEP 2: Ingesting Track Features")
    logger.info("=" * 60)

    try:
        recently_played = api_client.fetch_all_recently_played(max_items=1000)

        if not recently_played:
            logger.warning("No tracks found.")
            return

        track_ids = list(set([item['track']['id'] for item in recently_played
                              if 'track' in item and 'id' in item['track']]))
        logger.info(f"Found {len(track_ids)} unique tracks")

        logger.info("Fetching audio features...")
        try:
            audio_features = api_client.fetch_audio_features(track_ids)
        except RuntimeError as e:
            if "403" in str(e):
                logger.warning("⚠️  Audio features endpoint returned 403 - Missing Spotify API permissions")
                logger.warning("⚠️  Your Spotify app needs additional scopes. Skipping audio features...")
                audio_features = []
            else:
                raise

        logger.info("Fetching track metadata...")
        tracks_metadata = api_client.fetch_track_details(track_ids)

        logger.info(f"✅ Fetched {len(tracks_metadata)} tracks, {len(audio_features)} features")

        mapper = SpotifyMapper(spark)
        tracks_df = mapper.create_tracks_features_df(tracks_metadata, audio_features)

        DeltaWriter.write_tracks_features(
            tracks_df,
            config.paths.tracks_features,
            mode='append'
        )

        logger.info("✅ Track features ingestion complete\n")

    except Exception as e:
        logger.error(f"⚠️  Track features ingestion failed: {e}")
        logger.info("Continuing with remaining ingestion steps...\n")


def ingest_kaggle_dataset(spark: SparkSession, config) -> None:
    """Load Kaggle dataset with optimized processing."""
    logger.info("=" * 60)
    logger.info("STEP 3: Ingesting Kaggle Dataset")
    logger.info("=" * 60)
    
    if not config.paths.kaggle_csv.exists():
        logger.warning(f"Kaggle CSV not found. Skipping.")
        return
    
    try:
        logger.info(f"Loading Kaggle dataset from {config.paths.kaggle_csv}...")

        # Read CSV with optimized settings
        kaggle_df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "DROPMALFORMED")
            .csv(str(config.paths.kaggle_csv)))

        logger.info(f"✅ Loaded Kaggle dataset")

        # Map to Bronze schema
        mapper = KaggleMapper()
        kaggle_df_mapped = mapper.map_to_bronze_schema(kaggle_df)

        # Write to Bronze
        DeltaWriter.write_kaggle_tracks(
            kaggle_df_mapped,
            config.paths.kaggle_tracks,
            mode='overwrite'
        )

        logger.info("✅ Kaggle dataset ingestion complete\n")
        
    except Exception as e:
        logger.error(f"❌ Kaggle ingestion failed: {e}")
        logger.info("Continuing without Kaggle data...")


def run_pipeline():
    """Main pipeline execution."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"🚀 SPOTIFY ANALYTICS PIPELINE - Bronze Ingestion")
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        config = get_config()
        logger.info("✅ Configuration loaded")
        
        spark = create_spark_session(config.spark_app_name)
        
        logger.info("Initializing Spotify API client...")
        auth_client = SpotifyAuthClient(config.spotify)
        api_client = SpotifyAPIClient(config, auth_client)
        logger.info("✅ API client initialized\n")
        
        # Run ingestion (Spotify first - smaller data)
        ingest_listening_history(spark, config, api_client)
        ingest_tracks_features(spark, config, api_client)
        ingest_kaggle_dataset(spark, config)
        
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
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    sys.exit(run_pipeline())
