"""
DESCRIPTIVE ANALYTICS - Gold Layer

⚠️ ACADEMIC DISCLAIMER:
Descriptive analytics on SYNTHETIC audio features from single-user dataset.

CRITICAL LIMITATIONS:
1. Audio features are SYNTHETICALLY GENERATED (Spotify API 403 error)
2. Single-user dataset (n=~1,500) - patterns are anecdotal, not statistical
3. Describes what happened in the data, but data reflects synthetic randomness

WHAT THIS DEMONSTRATES:
✅ Descriptive analytics methodology and aggregation techniques
✅ Time-based pattern analysis
✅ Data summarization and visualization preparation
✅ SQL and Spark aggregation proficiency

WHAT THIS DOES NOT PROVIDE:
❌ Valid insights into actual music preferences
❌ Statistically significant behavioral patterns
❌ Generalizable user trends

See ACADEMIC_DISCLAIMER.md for full details.

Creates analytical tables that describe listening patterns:
1. Listening patterns by time (hour, day, part_of_day)
2. Top tracks by mood (valence, energy)
3. Genre preferences
4. Audio feature distributions

These tables answer: "What happened?" and "What is the current state?"
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, avg, stddev, min as spark_min, max as spark_max,
    sum as spark_sum, desc, asc, when, round as spark_round,
    row_number, dense_rank, percent_rank
)

from utils.logger import setup_logger

logger = setup_logger('descriptive_analytics')


class DescriptiveAnalytics:
    """Build descriptive analytics tables from Silver layer."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_silver_data(self, silver_path: str) -> DataFrame:
        """Load enriched listening data from Silver layer."""
        logger.info(f"Loading Silver layer data from: {silver_path}")
        try:
            df = self.spark.read.format('delta').load(silver_path)
            logger.info(f"✅ Loaded Silver layer data")
            return df
        except Exception as e:
            logger.error(f"Failed to load Silver data: {e}")
            raise

    def build_listening_patterns_by_time(self, df: DataFrame) -> DataFrame:
        """
        DESCRIPTIVE: Listening patterns aggregated by time dimensions.

        Answers: How many songs played per hour? Per day? Per part_of_day?
        """
        logger.info("Building listening patterns by time...")

        patterns = df.groupBy('hour_of_day', 'day_of_week', 'day_name', 'part_of_day', 'is_weekend') \
            .agg(
                count('*').alias('play_count'),
                avg('valence').alias('avg_valence'),
                avg('energy').alias('avg_energy'),
                avg('danceability').alias('avg_danceability'),
                avg('acousticness').alias('avg_acousticness'),
                avg('tempo').alias('avg_tempo')
            ) \
            .withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
            .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
            .withColumn('avg_danceability', spark_round(col('avg_danceability'), 3)) \
            .withColumn('avg_acousticness', spark_round(col('avg_acousticness'), 3)) \
            .withColumn('avg_tempo', spark_round(col('avg_tempo'), 1)) \
            .orderBy('hour_of_day', 'day_of_week')

        logger.info(f"✅ Created listening patterns table")
        return patterns

    def build_top_tracks_by_mood(self, df: DataFrame, top_n: int = 50) -> DataFrame:
        """
        DESCRIPTIVE: Top tracks categorized by mood (high valence, high energy, etc.)

        Answers: What are the happiest songs? Most energetic? Most calm?
        """
        logger.info(f"Building top {top_n} tracks by mood categories...")

        # Define mood categories
        mood_categories = df.withColumn(
            'mood_category',
            when((col('valence') >= 0.6) & (col('energy') >= 0.6), 'Happy_Energetic')
            .when((col('valence') >= 0.6) & (col('energy') < 0.4), 'Happy_Calm')
            .when((col('valence') < 0.4) & (col('energy') >= 0.6), 'Sad_Energetic')
            .when((col('valence') < 0.4) & (col('energy') < 0.4), 'Sad_Calm')
            .otherwise('Neutral')
        )

        # Count plays per track
        track_stats = mood_categories.groupBy(
            'track_id', 'track_name', 'artist_name', 'mood_category',
            'valence', 'energy', 'danceability', 'tempo'
        ).agg(
            count('*').alias('play_count')
        )

        # Rank within each mood category
        window = Window.partitionBy('mood_category').orderBy(desc('play_count'))

        top_tracks = track_stats.withColumn('rank_in_category', row_number().over(window)) \
            .filter(col('rank_in_category') <= top_n) \
            .orderBy('mood_category', 'rank_in_category')

        logger.info(f"✅ Created top tracks by mood table")
        return top_tracks

    def build_audio_feature_distributions(self, df: DataFrame) -> DataFrame:
        """
        DESCRIPTIVE: Statistical distributions of audio features.

        Answers: What's the average valence? Energy spread? Tempo range?
        """
        logger.info("Building audio feature distributions...")

        features = ['valence', 'energy', 'danceability', 'acousticness',
                   'instrumentalness', 'speechiness', 'loudness', 'tempo']

        # Calculate statistics for each feature
        distributions = []

        for feature in features:
            stats = df.agg(
                avg(feature).alias('mean'),
                stddev(feature).alias('std_dev'),
                spark_min(feature).alias('min'),
                spark_max(feature).alias('max')
            ).collect()[0]

            distributions.append({
                'feature_name': feature,
                'mean': round(float(stats['mean']) if stats['mean'] else 0, 3),
                'std_dev': round(float(stats['std_dev']) if stats['std_dev'] else 0, 3),
                'min': round(float(stats['min']) if stats['min'] else 0, 3),
                'max': round(float(stats['max']) if stats['max'] else 0, 3)
            })

        distributions_df = self.spark.createDataFrame(distributions)

        logger.info(f"✅ Created audio feature distributions table")
        return distributions_df

    def build_temporal_trends(self, df: DataFrame) -> DataFrame:
        """
        DESCRIPTIVE: Trends over time (daily aggregations).

        Answers: How do listening habits change day by day?
        """
        logger.info("Building temporal trends...")

        trends = df.withColumn('date', col('played_at').cast('date')) \
            .groupBy('date') \
            .agg(
                count('*').alias('total_plays'),
                avg('valence').alias('avg_valence'),
                avg('energy').alias('avg_energy'),
                avg('tempo').alias('avg_tempo'),
                count(when(col('is_weekend'), 1)).alias('weekend_plays')
            ) \
            .withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
            .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
            .withColumn('avg_tempo', spark_round(col('avg_tempo'), 1)) \
            .orderBy('date')

        logger.info(f"✅ Created temporal trends table")
        return trends

    def build_feature_source_coverage(self, df: DataFrame) -> DataFrame:
        """
        DESCRIPTIVE: Data quality metrics - feature source coverage.

        Answers: How much data is real vs synthetic vs Kaggle?
        """
        logger.info("Building feature source coverage metrics...")

        coverage = df.groupBy('feature_source') \
            .agg(
                count('*').alias('record_count')
            ) \
            .withColumn('total', spark_sum('record_count').over(Window.partitionBy())) \
            .withColumn('percentage', spark_round((col('record_count') / col('total')) * 100, 2)) \
            .select('feature_source', 'record_count', 'percentage') \
            .orderBy(desc('record_count'))

        logger.info(f"✅ Created feature source coverage table")
        return coverage

    def write_gold_table(self, df: DataFrame, table_name: str, gold_path: str):
        """Write Gold layer table to Delta Lake."""
        output_path = f"{gold_path}/descriptive/{table_name}"
        logger.info(f"Writing {table_name} to: {output_path}")

        try:
            df.write \
                .format('delta') \
                .mode('overwrite') \
                .save(output_path)

            logger.info(f"✅ Successfully wrote {table_name}")
        except Exception as e:
            logger.error(f"Failed to write {table_name}: {e}")
            raise

    def build_all(self, silver_path: str, gold_path: str):
        """Build all descriptive analytics tables."""
        logger.info("=" * 80)
        logger.info("BUILDING DESCRIPTIVE ANALYTICS - GOLD LAYER")
        logger.info("=" * 80)

        # Load Silver data
        silver_df = self.load_silver_data(silver_path)

        # Build all descriptive tables
        tables = {
            'listening_patterns_by_time': self.build_listening_patterns_by_time(silver_df),
            'top_tracks_by_mood': self.build_top_tracks_by_mood(silver_df, top_n=50),
            'audio_feature_distributions': self.build_audio_feature_distributions(silver_df),
            'temporal_trends': self.build_temporal_trends(silver_df),
            'feature_source_coverage': self.build_feature_source_coverage(silver_df),
        }

        # Write all tables
        for table_name, table_df in tables.items():
            self.write_gold_table(table_df, table_name, gold_path)

        logger.info("=" * 80)
        logger.info("✅ DESCRIPTIVE ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Created {len(tables)} analytical tables:")
        for table_name in tables.keys():
            logger.info(f"  - {table_name}")


def main():
    """Main execution."""
    logger.info("Starting Descriptive Analytics build...")

    # Paths
    silver_path = '/app/data/silver/listening_with_features'
    gold_path = '/app/data/gold'

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("DescriptiveAnalytics") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        analytics = DescriptiveAnalytics(spark)
        analytics.build_all(silver_path, gold_path)

        logger.info("=" * 80)
        logger.info("✅ Descriptive Analytics build SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Descriptive Analytics build FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
