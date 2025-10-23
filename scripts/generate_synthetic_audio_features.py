"""
Generate deterministic synthetic audio features for tracks.

Uses hash-based seeding per track_id to ensure reproducible results across runs.
Writes to a separate Bronze table with source='synthetic' flag.

Usage:
    python3 scripts/generate_synthetic_audio_features.py \
        --history-path /app/data/bronze/listening_history_bronze \
        --out-path /app/data/bronze/my_tracks_features_bronze_synthetic \
        --limit 1000
"""

import argparse
import hashlib
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    IntegerType, BooleanType, TimestampType
)

from utils.logger import setup_logger

logger = setup_logger('synthetic_features')


class SyntheticAudioFeatureGenerator:
    """Generate deterministic synthetic audio features using hash-based seeding."""

    # Feature ranges based on Spotify API documentation
    FEATURE_RANGES = {
        'danceability': (0.15, 0.85),
        'energy': (0.10, 0.90),
        'valence': (0.10, 0.85),
        'acousticness': (0.05, 0.95),
        'instrumentalness': (0.00, 0.80),
        'speechiness': (0.03, 0.50),
        'liveness': (0.05, 0.40),
        'loudness': (-25.0, -3.0),
        'tempo': (60.0, 180.0),
    }

    DISCRETE_RANGES = {
        'time_signature': [3, 4, 5],  # Most common time signatures
        'key': list(range(0, 12)),     # 0 = C, 1 = C♯/D♭, etc.
        'mode': [0, 1],                # 0 = minor, 1 = major
    }

    def __init__(self):
        pass

    def _get_deterministic_seed(self, track_id: str) -> int:
        """Generate deterministic seed from track_id using SHA256 hash."""
        hash_bytes = hashlib.sha256(track_id.encode('utf-8')).digest()
        # Use first 8 bytes to create integer seed
        seed = int.from_bytes(hash_bytes[:8], byteorder='big')
        return seed

    def _generate_features_for_track(self, track_id: str) -> dict:
        """Generate all synthetic audio features for a single track_id."""
        # Set deterministic seed based on track_id
        seed = self._get_deterministic_seed(track_id)
        rng = random.Random(seed)

        features = {'track_id': track_id}

        # Generate continuous features
        for feature_name, (min_val, max_val) in self.FEATURE_RANGES.items():
            # Use beta distribution for more realistic values (concentrated away from extremes)
            # Beta(2, 2) gives bell-shaped distribution between 0 and 1
            normalized = rng.betavariate(2, 2)
            value = min_val + normalized * (max_val - min_val)
            features[feature_name] = round(value, 4)

        # Generate discrete features
        features['time_signature'] = rng.choice(self.DISCRETE_RANGES['time_signature'])
        features['key'] = rng.choice(self.DISCRETE_RANGES['key'])
        features['mode'] = rng.choice(self.DISCRETE_RANGES['mode'])

        # Metadata flags
        features['source'] = 'synthetic'
        features['_generated_at'] = datetime.now()

        return features

    def generate_synthetic_schema(self) -> StructType:
        """Define schema for synthetic audio features table."""
        return StructType([
            StructField("track_id", StringType(), nullable=False),

            # Audio features (continuous)
            StructField("danceability", FloatType(), nullable=False),
            StructField("energy", FloatType(), nullable=False),
            StructField("valence", FloatType(), nullable=False),
            StructField("acousticness", FloatType(), nullable=False),
            StructField("instrumentalness", FloatType(), nullable=False),
            StructField("speechiness", FloatType(), nullable=False),
            StructField("liveness", FloatType(), nullable=False),
            StructField("loudness", FloatType(), nullable=False),
            StructField("tempo", FloatType(), nullable=False),

            # Audio features (discrete)
            StructField("time_signature", IntegerType(), nullable=False),
            StructField("key", IntegerType(), nullable=False),
            StructField("mode", IntegerType(), nullable=False),

            # Metadata
            StructField("source", StringType(), nullable=False),
            StructField("_generated_at", TimestampType(), nullable=False),
        ])

    def generate_features_for_tracks(self, track_ids: List[str]) -> List[dict]:
        """Generate synthetic features for a list of track IDs."""
        logger.info(f"Generating synthetic features for {len(track_ids)} tracks...")

        features_list = []
        for i, track_id in enumerate(track_ids):
            features = self._generate_features_for_track(track_id)
            features_list.append(features)

            if (i + 1) % 100 == 0:
                logger.debug(f"Generated features for {i + 1}/{len(track_ids)} tracks")

        logger.info(f"✅ Generated {len(features_list)} synthetic feature records")
        return features_list


def get_distinct_track_ids(spark: SparkSession, history_path: str, limit: int = None) -> List[str]:
    """Extract distinct track_ids from listening history."""
    logger.info(f"Reading listening history from: {history_path}")

    try:
        history_df = spark.read.format('delta').load(history_path)

        # Get distinct track IDs
        distinct_tracks = history_df.select('track_id').distinct()

        if limit:
            distinct_tracks = distinct_tracks.limit(limit)

        track_ids = [row.track_id for row in distinct_tracks.collect()]

        logger.info(f"✅ Found {len(track_ids)} distinct track IDs")
        return track_ids

    except Exception as e:
        logger.error(f"Failed to read listening history: {e}")
        raise


def write_synthetic_features(spark: SparkSession, features: List[dict],
                             out_path: str, schema: StructType):
    """Write synthetic features to Delta Lake."""
    logger.info(f"Writing {len(features)} synthetic features to: {out_path}")

    try:
        # Create DataFrame from generated features
        df = spark.createDataFrame(features, schema=schema)

        # Write to Delta Lake (overwrite mode for idempotency)
        df.write \
            .format('delta') \
            .mode('overwrite') \
            .save(out_path)

        logger.info(f"✅ Successfully wrote synthetic features to Delta Lake")

    except Exception as e:
        logger.error(f"Failed to write synthetic features: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Generate deterministic synthetic audio features for tracks'
    )
    parser.add_argument(
        '--history-path',
        type=str,
        default='/app/data/bronze/listening_history_bronze',
        help='Path to listening history Bronze table'
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default='/app/data/bronze/my_tracks_features_bronze_synthetic',
        help='Output path for synthetic features table'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tracks to process (for testing)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SYNTHETIC AUDIO FEATURES GENERATOR")
    logger.info("=" * 80)
    logger.info(f"History path: {args.history_path}")
    logger.info(f"Output path: {args.out_path}")
    logger.info(f"Limit: {args.limit if args.limit else 'None (process all)'}")
    logger.info("=" * 80)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("SyntheticAudioFeatures") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        # Step 1: Get distinct track IDs from listening history
        track_ids = get_distinct_track_ids(spark, args.history_path, args.limit)

        if not track_ids:
            logger.warning("⚠️  No track IDs found in listening history")
            return

        # Step 2: Generate synthetic features
        generator = SyntheticAudioFeatureGenerator()
        features = generator.generate_features_for_tracks(track_ids)

        # Step 3: Write to Delta Lake
        schema = generator.generate_synthetic_schema()
        write_synthetic_features(spark, features, args.out_path, schema)

        # Step 4: Summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Distinct tracks processed: {len(track_ids)}")
        logger.info(f"✅ Synthetic features generated: {len(features)}")
        logger.info(f"✅ Output location: {args.out_path}")
        logger.info("=" * 80)
        logger.info("✅ Synthetic feature generation complete!")

    except Exception as e:
        logger.error(f"❌ Synthetic feature generation failed: {e}")
        raise

    finally:
        spark.stop()


if __name__ == '__main__':
    main()
