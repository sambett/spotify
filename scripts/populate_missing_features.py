"""
Automatic synthetic feature populator for missing track_ids.

This script:
1. Finds track_ids in listening history that lack audio features
2. Checks if they already have real features (my_tracks_features_bronze)
3. Checks if they already have synthetic features (my_tracks_features_bronze_synthetic)
4. Generates synthetic features for remaining track_ids
5. Appends them to the synthetic features table

This is designed to run automatically after listening history ingestion.

Usage:
    python3 scripts/populate_missing_features.py \
        --history-path /app/data/bronze/listening_history_bronze \
        --real-features-path /app/data/bronze/my_tracks_features_bronze \
        --synthetic-features-path /app/data/bronze/my_tracks_features_bronze_synthetic \
        --allow-synthetic true
"""

import argparse
import hashlib
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    IntegerType, TimestampType
)

from utils.logger import setup_logger

logger = setup_logger('feature_populator')


class MissingFeaturePopulator:
    """Automatically populate missing audio features with synthetic data."""

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
        'time_signature': [3, 4, 5],
        'key': list(range(0, 12)),
        'mode': [0, 1],
    }

    def __init__(self, spark: SparkSession, allow_synthetic: bool = True):
        self.spark = spark
        self.allow_synthetic = allow_synthetic

    def get_track_ids_from_history(self, history_path: str) -> Set[str]:
        """Get all distinct track_ids from listening history."""
        logger.info(f"Reading track IDs from listening history: {history_path}")
        try:
            df = self.spark.read.format('delta').load(history_path)
            track_ids = {row.track_id for row in df.select('track_id').distinct().collect()}
            logger.info(f"✅ Found {len(track_ids)} distinct tracks in listening history")
            return track_ids
        except Exception as e:
            logger.error(f"Failed to read listening history: {e}")
            raise

    def get_track_ids_with_real_features(self, real_path: str) -> Set[str]:
        """Get track_ids that have real (non-NULL) audio features."""
        logger.info(f"Reading track IDs with real features: {real_path}")
        try:
            df = self.spark.read.format('delta').load(real_path)

            # Filter to tracks with at least one non-null audio feature
            df_with_features = df.filter(
                (col('danceability').isNotNull()) |
                (col('energy').isNotNull()) |
                (col('valence').isNotNull())
            )

            track_ids = {row.track_id for row in df_with_features.select('track_id').collect()}
            logger.info(f"✅ Found {len(track_ids)} tracks with real features")
            return track_ids
        except Exception as e:
            logger.warning(f"⚠️  Could not read real features (may not exist yet): {e}")
            return set()

    def get_track_ids_with_synthetic_features(self, synthetic_path: str) -> Set[str]:
        """Get track_ids that already have synthetic features."""
        logger.info(f"Reading track IDs with synthetic features: {synthetic_path}")
        try:
            df = self.spark.read.format('delta').load(synthetic_path)
            track_ids = {row.track_id for row in df.select('track_id').collect()}
            logger.info(f"✅ Found {len(track_ids)} tracks with synthetic features")
            return track_ids
        except Exception as e:
            logger.warning(f"⚠️  Could not read synthetic features (may not exist yet): {e}")
            return set()

    def find_missing_track_ids(self, history_path: str, real_path: str,
                               synthetic_path: str) -> List[str]:
        """Find track_ids that need synthetic features."""
        logger.info("=" * 80)
        logger.info("IDENTIFYING TRACKS NEEDING FEATURES")
        logger.info("=" * 80)

        all_tracks = self.get_track_ids_from_history(history_path)
        real_tracks = self.get_track_ids_with_real_features(real_path)
        synthetic_tracks = self.get_track_ids_with_synthetic_features(synthetic_path)

        # Missing = tracks in history but not in real OR synthetic
        covered_tracks = real_tracks | synthetic_tracks
        missing_tracks = all_tracks - covered_tracks

        logger.info("=" * 80)
        logger.info("COVERAGE ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Total tracks in history:    {len(all_tracks)}")
        logger.info(f"Tracks with real features:  {len(real_tracks)}")
        logger.info(f"Tracks with synthetic:      {len(synthetic_tracks)}")
        logger.info(f"Total covered:              {len(covered_tracks)}")
        logger.info(f"Missing (need synthetic):   {len(missing_tracks)}")

        if all_tracks:
            coverage_pct = (len(covered_tracks) / len(all_tracks)) * 100
            logger.info(f"Current coverage:           {coverage_pct:.1f}%")

        logger.info("=" * 80)

        return sorted(list(missing_tracks))

    def _get_deterministic_seed(self, track_id: str) -> int:
        """Generate deterministic seed from track_id using SHA256 hash."""
        hash_bytes = hashlib.sha256(track_id.encode('utf-8')).digest()
        seed = int.from_bytes(hash_bytes[:8], byteorder='big')
        return seed

    def _generate_features_for_track(self, track_id: str) -> dict:
        """Generate all synthetic audio features for a single track_id."""
        seed = self._get_deterministic_seed(track_id)
        rng = random.Random(seed)

        features = {'track_id': track_id}

        # Generate continuous features
        for feature_name, (min_val, max_val) in self.FEATURE_RANGES.items():
            normalized = rng.betavariate(2, 2)
            value = min_val + normalized * (max_val - min_val)
            features[feature_name] = round(value, 4)

        # Generate discrete features
        features['time_signature'] = rng.choice(self.DISCRETE_RANGES['time_signature'])
        features['key'] = rng.choice(self.DISCRETE_RANGES['key'])
        features['mode'] = rng.choice(self.DISCRETE_RANGES['mode'])

        # Metadata
        features['source'] = 'synthetic'
        features['_generated_at'] = datetime.now()

        return features

    def generate_features(self, track_ids: List[str]) -> List[dict]:
        """Generate synthetic features for multiple track IDs."""
        if not track_ids:
            logger.info("No tracks to generate features for")
            return []

        logger.info(f"Generating synthetic features for {len(track_ids)} tracks...")

        features_list = []
        for i, track_id in enumerate(track_ids):
            features = self._generate_features_for_track(track_id)
            features_list.append(features)

            if (i + 1) % 100 == 0:
                logger.debug(f"Generated {i + 1}/{len(track_ids)} features")

        logger.info(f"✅ Generated {len(features_list)} synthetic feature records")
        return features_list

    def get_synthetic_schema(self) -> StructType:
        """Define schema for synthetic audio features."""
        return StructType([
            StructField("track_id", StringType(), nullable=False),
            StructField("danceability", FloatType(), nullable=False),
            StructField("energy", FloatType(), nullable=False),
            StructField("valence", FloatType(), nullable=False),
            StructField("acousticness", FloatType(), nullable=False),
            StructField("instrumentalness", FloatType(), nullable=False),
            StructField("speechiness", FloatType(), nullable=False),
            StructField("liveness", FloatType(), nullable=False),
            StructField("loudness", FloatType(), nullable=False),
            StructField("tempo", FloatType(), nullable=False),
            StructField("time_signature", IntegerType(), nullable=False),
            StructField("key", IntegerType(), nullable=False),
            StructField("mode", IntegerType(), nullable=False),
            StructField("source", StringType(), nullable=False),
            StructField("_generated_at", TimestampType(), nullable=False),
        ])

    def append_synthetic_features(self, features: List[dict], synthetic_path: str):
        """Append new synthetic features to Delta table (idempotent)."""
        if not features:
            logger.info("No features to write")
            return

        logger.info(f"Writing {len(features)} new synthetic features to: {synthetic_path}")

        try:
            schema = self.get_synthetic_schema()
            df = self.spark.createDataFrame(features, schema=schema)

            # Use append mode (idempotent: same track_id will have same features due to deterministic generation)
            df.write \
                .format('delta') \
                .mode('append') \
                .save(synthetic_path)

            logger.info(f"✅ Successfully appended {len(features)} synthetic features")

        except Exception as e:
            logger.error(f"Failed to write synthetic features: {e}")
            raise

    def populate(self, history_path: str, real_path: str, synthetic_path: str):
        """Main population workflow."""
        logger.info("=" * 80)
        logger.info("AUTOMATIC FEATURE POPULATION")
        logger.info("=" * 80)
        logger.info(f"Allow synthetic: {self.allow_synthetic}")
        logger.info("=" * 80)

        if not self.allow_synthetic:
            logger.warning("⚠️  ALLOW_SYNTHETIC=false: Skipping synthetic feature generation")
            logger.info("Pipeline will continue with existing features only")
            return

        # Step 1: Find tracks needing features
        missing_track_ids = self.find_missing_track_ids(history_path, real_path, synthetic_path)

        if not missing_track_ids:
            logger.info("✅ All tracks already have features (real or synthetic)")
            return

        # Step 2: Generate synthetic features for missing tracks
        features = self.generate_features(missing_track_ids)

        # Step 3: Append to synthetic features table
        self.append_synthetic_features(features, synthetic_path)

        # Step 4: Summary
        logger.info("=" * 80)
        logger.info("POPULATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ New tracks detected:      {len(missing_track_ids)}")
        logger.info(f"✅ Synthetic features added: {len(features)}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Automatically populate missing audio features with synthetic data'
    )
    parser.add_argument(
        '--history-path',
        type=str,
        default='/app/data/bronze/listening_history_bronze',
        help='Path to listening history Bronze table'
    )
    parser.add_argument(
        '--real-features-path',
        type=str,
        default='/app/data/bronze/my_tracks_features_bronze',
        help='Path to real Spotify API features'
    )
    parser.add_argument(
        '--synthetic-features-path',
        type=str,
        default='/app/data/bronze/my_tracks_features_bronze_synthetic',
        help='Path to synthetic features table'
    )
    parser.add_argument(
        '--allow-synthetic',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='Enable/disable synthetic feature generation (kill-switch)'
    )

    args = parser.parse_args()

    allow_synthetic = args.allow_synthetic.lower() == 'true'

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("MissingFeaturePopulator") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        populator = MissingFeaturePopulator(spark, allow_synthetic=allow_synthetic)

        populator.populate(
            history_path=args.history_path,
            real_path=args.real_features_path,
            synthetic_path=args.synthetic_features_path
        )

        logger.info("=" * 80)
        logger.info("✅ Feature population complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Feature population failed: {e}")
        raise

    finally:
        spark.stop()


if __name__ == '__main__':
    main()
