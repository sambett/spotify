"""
Build Silver layer: listening history enriched with audio features.

Preference order:
1. Real Spotify API features (if available)
2. Synthetic features (fallback)
3. Kaggle features (additional fallback)

Includes feature_source column to track data provenance.

Usage:
    python3 scripts/build_silver_listening_with_features.py \
        --history-path /app/data/bronze/listening_history_bronze \
        --real-features-path /app/data/bronze/my_tracks_features_bronze \
        --synthetic-features-path /app/data/bronze/my_tracks_features_bronze_synthetic \
        --kaggle-features-path /app/data/bronze/kaggle_tracks_bronze \
        --out-path /app/data/silver/listening_with_features
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, coalesce, lit, when, hour, dayofweek,
    date_format, current_timestamp
)

from utils.logger import setup_logger

logger = setup_logger('silver_builder')


class SilverLayerBuilder:
    """Build Silver layer with intelligent feature source preference."""

    AUDIO_FEATURES = [
        'danceability', 'energy', 'valence', 'acousticness',
        'instrumentalness', 'speechiness', 'liveness',
        'loudness', 'tempo', 'time_signature', 'key', 'mode'
    ]

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_listening_history(self, path: str) -> DataFrame:
        """Load listening history from Bronze layer."""
        logger.info(f"Loading listening history from: {path}")
        try:
            df = self.spark.read.format('delta').load(path)
            count = df.count()
            logger.info(f"✅ Loaded {count} listening events")
            return df
        except Exception as e:
            logger.error(f"Failed to load listening history: {e}")
            raise

    def load_real_features(self, path: str) -> DataFrame:
        """Load real Spotify API features (may have NULLs)."""
        logger.info(f"Loading real features from: {path}")
        try:
            df = self.spark.read.format('delta').load(path)

            # Filter to only tracks that have at least one non-null audio feature
            feature_cols = [col(f) for f in self.AUDIO_FEATURES if f in df.columns]
            if feature_cols:
                df = df.filter(
                    coalesce(*feature_cols).isNotNull()
                )

            count = df.count()
            logger.info(f"✅ Loaded {count} tracks with real audio features")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Could not load real features: {e}")
            return None

    def load_synthetic_features(self, path: str) -> DataFrame:
        """Load synthetic features."""
        logger.info(f"Loading synthetic features from: {path}")
        try:
            df = self.spark.read.format('delta').load(path)
            count = df.count()
            logger.info(f"✅ Loaded {count} tracks with synthetic features")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Could not load synthetic features: {e}")
            return None

    def load_kaggle_features(self, path: str) -> DataFrame:
        """Load Kaggle dataset features."""
        logger.info(f"Loading Kaggle features from: {path}")
        try:
            df = self.spark.read.format('delta').load(path)
            count = df.count()
            logger.info(f"✅ Loaded {count} tracks from Kaggle")
            return df
        except Exception as e:
            logger.warning(f"⚠️  Could not load Kaggle features: {e}")
            return None

    def merge_features_with_preference(self, listening: DataFrame,
                                      real: DataFrame = None,
                                      synthetic: DataFrame = None,
                                      kaggle: DataFrame = None) -> DataFrame:
        """
        Merge listening history with audio features using preference order:
        real > synthetic > kaggle
        """
        logger.info("Merging features with preference order: real → synthetic → kaggle")

        result = listening

        # Prepare feature sources for joining
        sources = []

        if real is not None:
            real_features = real.select(
                col('track_id'),
                *[col(f).alias(f'{f}_real') for f in self.AUDIO_FEATURES if f in real.columns]
            )
            sources.append(('real', real_features))

        if synthetic is not None:
            synthetic_features = synthetic.select(
                col('track_id'),
                *[col(f).alias(f'{f}_synthetic') for f in self.AUDIO_FEATURES if f in synthetic.columns]
            )
            sources.append(('synthetic', synthetic_features))

        if kaggle is not None:
            kaggle_features = kaggle.select(
                col('track_id'),
                *[col(f).alias(f'{f}_kaggle') for f in self.AUDIO_FEATURES if f in kaggle.columns]
            )
            sources.append(('kaggle', kaggle_features))

        # Join all sources
        for source_name, source_df in sources:
            result = result.join(source_df, on='track_id', how='left')

        # Apply preference logic: coalesce(real, synthetic, kaggle)
        for feature in self.AUDIO_FEATURES:
            available_cols = []

            if real is not None and f'{feature}_real' in result.columns:
                available_cols.append(col(f'{feature}_real'))
            if synthetic is not None and f'{feature}_synthetic' in result.columns:
                available_cols.append(col(f'{feature}_synthetic'))
            if kaggle is not None and f'{feature}_kaggle' in result.columns:
                available_cols.append(col(f'{feature}_kaggle'))

            if available_cols:
                result = result.withColumn(feature, coalesce(*available_cols))

        # Add feature_source column to track provenance
        source_conditions = []
        if real is not None:
            source_conditions.append(
                (col('danceability_real').isNotNull(), 'real')
            )
        if synthetic is not None:
            source_conditions.append(
                (col('danceability_synthetic').isNotNull(), 'synthetic')
            )
        if kaggle is not None:
            source_conditions.append(
                (col('danceability_kaggle').isNotNull(), 'kaggle')
            )

        # Build when-otherwise chain
        feature_source_col = lit('unknown')
        for condition, source_name in reversed(source_conditions):
            feature_source_col = when(condition, source_name).otherwise(feature_source_col)

        result = result.withColumn('feature_source', feature_source_col)

        # Drop temporary columns (_real, _synthetic, _kaggle suffixes)
        temp_cols = [c for c in result.columns if c.endswith(('_real', '_synthetic', '_kaggle'))]
        result = result.drop(*temp_cols)

        return result

    def add_time_dimensions(self, df: DataFrame) -> DataFrame:
        """Add time-based dimensions for analysis."""
        logger.info("Adding time dimensions...")

        df = df.withColumn('hour_of_day', hour(col('played_at')))
        df = df.withColumn('day_of_week', dayofweek(col('played_at')))
        df = df.withColumn('day_name', date_format(col('played_at'), 'EEEE'))
        df = df.withColumn('is_weekend',
                          when(col('day_of_week').isin([1, 7]), True).otherwise(False))

        # Part of day classification
        df = df.withColumn('part_of_day',
                          when(col('hour_of_day').between(6, 11), 'morning')
                          .when(col('hour_of_day').between(12, 17), 'afternoon')
                          .when(col('hour_of_day').between(18, 22), 'evening')
                          .otherwise('night'))

        logger.info("✅ Time dimensions added")
        return df

    def add_metadata(self, df: DataFrame) -> DataFrame:
        """Add Silver layer metadata."""
        return df.withColumn('_silver_created_at', current_timestamp())

    def build_silver_layer(self, history_path: str,
                          real_path: str = None,
                          synthetic_path: str = None,
                          kaggle_path: str = None) -> DataFrame:
        """Build complete Silver layer."""
        logger.info("=" * 80)
        logger.info("BUILDING SILVER LAYER")
        logger.info("=" * 80)

        # Load all data sources
        listening = self.load_listening_history(history_path)

        real = self.load_real_features(real_path) if real_path else None
        synthetic = self.load_synthetic_features(synthetic_path) if synthetic_path else None
        kaggle = self.load_kaggle_features(kaggle_path) if kaggle_path else None

        # Merge features
        enriched = self.merge_features_with_preference(listening, real, synthetic, kaggle)

        # Add time dimensions
        enriched = self.add_time_dimensions(enriched)

        # Add metadata
        enriched = self.add_metadata(enriched)

        return enriched

    def write_silver_layer(self, df: DataFrame, out_path: str):
        """Write Silver layer to Delta Lake."""
        logger.info(f"Writing Silver layer to: {out_path}")

        try:
            df.write \
                .format('delta') \
                .mode('overwrite') \
                .partitionBy('date') \
                .save(out_path)

            logger.info("✅ Silver layer written successfully")
        except Exception as e:
            logger.error(f"Failed to write Silver layer: {e}")
            raise

    def log_coverage_statistics(self, df: DataFrame):
        """Log feature coverage statistics."""
        logger.info("=" * 80)
        logger.info("FEATURE COVERAGE STATISTICS")
        logger.info("=" * 80)

        total_count = df.count()

        # Count by feature source
        source_counts = df.groupBy('feature_source').count().collect()
        for row in source_counts:
            source = row['feature_source']
            count = row['count']
            percentage = (count / total_count * 100) if total_count > 0 else 0
            logger.info(f"{source:12s}: {count:6d} ({percentage:5.1f}%)")

        # Count records with non-null features
        if 'valence' in df.columns:
            feature_count = df.filter(col('valence').isNotNull()).count()
            feature_percentage = (feature_count / total_count * 100) if total_count > 0 else 0
            logger.info(f"{'Total with features':12s}: {feature_count:6d} ({feature_percentage:5.1f}%)")

        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Build Silver layer with intelligent feature merging'
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
        help='Path to synthetic features'
    )
    parser.add_argument(
        '--kaggle-features-path',
        type=str,
        default='/app/data/bronze/kaggle_tracks_bronze',
        help='Path to Kaggle dataset'
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default='/app/data/silver/listening_with_features',
        help='Output path for Silver layer'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SILVER LAYER BUILDER")
    logger.info("=" * 80)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("SilverLayerBuilder") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        builder = SilverLayerBuilder(spark)

        # Build Silver layer
        silver_df = builder.build_silver_layer(
            history_path=args.history_path,
            real_path=args.real_features_path,
            synthetic_path=args.synthetic_features_path,
            kaggle_path=args.kaggle_features_path
        )

        # Log coverage statistics
        builder.log_coverage_statistics(silver_df)

        # Write Silver layer
        builder.write_silver_layer(silver_df, args.out_path)

        logger.info("=" * 80)
        logger.info("✅ Silver layer build complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Silver layer build failed: {e}")
        raise

    finally:
        spark.stop()


if __name__ == '__main__':
    main()
