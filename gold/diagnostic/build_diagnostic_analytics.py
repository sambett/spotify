"""
DIAGNOSTIC ANALYTICS - Gold Layer

⚠️ ACADEMIC DISCLAIMER:
Diagnostic analysis on SYNTHETIC audio features - causality claims are NOT valid.

CRITICAL LIMITATIONS:
1. Audio features are SYNTHETICALLY GENERATED (Spotify API 403 error)
2. Single-user dataset (n=~1,500) prevents causal inference
3. Correlations on synthetic data do not represent real relationships
4. Cannot establish "why" with anecdotal single-user data

WHAT THIS DEMONSTRATES:
✅ Diagnostic analytics methodology (correlation analysis)
✅ Time-series analysis techniques
✅ Feature relationship exploration
✅ Root cause analysis frameworks

WHAT THIS DOES NOT PROVIDE:
❌ Valid causal explanations for listening behavior
❌ Statistically significant correlations
❌ Generalizable behavioral insights

See ACADEMIC_DISCLAIMER.md for full details.

Creates analytical tables that explain WHY patterns occur:
1. Mood correlations with time of day
2. Feature correlations (e.g., valence vs energy)
3. Listening triggers (what causes spikes in certain moods)
4. Behavioral patterns (weekend vs weekday differences)

These tables answer: "Why did this happen?" and "What are the root causes?"
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, avg, stddev, corr, when, desc,
    round as spark_round, lag, lead, abs as spark_abs
)

from utils.logger import setup_logger

logger = setup_logger('diagnostic_analytics')


class DiagnosticAnalytics:
    """Build diagnostic analytics tables from Silver layer."""

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

    def build_mood_time_correlations(self, df: DataFrame) -> DataFrame:
        """
        DIAGNOSTIC: Correlations between mood features and time of day.

        Answers: Why do we listen to happy music in mornings? Sad music at night?
        """
        logger.info("Building mood-time correlations...")

        # Calculate average mood by hour
        mood_by_hour = df.groupBy('hour_of_day').agg(
            avg('valence').alias('avg_valence'),
            avg('energy').alias('avg_energy'),
            avg('danceability').alias('avg_danceability'),
            avg('acousticness').alias('avg_acousticness'),
            count('*').alias('sample_size')
        ).orderBy('hour_of_day')

        # Add variance to show spread
        mood_variance = df.groupBy('hour_of_day').agg(
            stddev('valence').alias('valence_stddev'),
            stddev('energy').alias('energy_stddev')
        )

        correlations = mood_by_hour.join(mood_variance, 'hour_of_day') \
            .withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
            .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
            .withColumn('avg_danceability', spark_round(col('avg_danceability'), 3)) \
            .withColumn('avg_acousticness', spark_round(col('avg_acousticness'), 3)) \
            .withColumn('valence_stddev', spark_round(col('valence_stddev'), 3)) \
            .withColumn('energy_stddev', spark_round(col('energy_stddev'), 3)) \
            .orderBy('hour_of_day')

        logger.info(f"✅ Created mood-time correlations table")
        return correlations

    def build_feature_correlations(self, df: DataFrame) -> DataFrame:
        """
        DIAGNOSTIC: Correlation matrix between audio features.

        Answers: Why do high-energy songs also have high danceability?
        """
        logger.info("Building feature correlation matrix...")

        features = ['valence', 'energy', 'danceability', 'acousticness',
                   'instrumentalness', 'tempo', 'loudness']

        correlations = []

        for i, feature1 in enumerate(features):
            for feature2 in features[i+1:]:  # Only upper triangle
                corr_value = df.select(corr(feature1, feature2)).collect()[0][0]

                correlations.append({
                    'feature_1': feature1,
                    'feature_2': feature2,
                    'correlation': round(float(corr_value) if corr_value else 0, 3),
                    'strength': (
                        'Strong' if abs(corr_value or 0) > 0.7
                        else 'Moderate' if abs(corr_value or 0) > 0.4
                        else 'Weak'
                    )
                })

        corr_df = self.spark.createDataFrame(correlations) \
            .orderBy(desc('correlation'))

        logger.info(f"✅ Created feature correlations table")
        return corr_df

    def build_weekend_vs_weekday_analysis(self, df: DataFrame) -> DataFrame:
        """
        DIAGNOSTIC: Compare listening behavior on weekends vs weekdays.

        Answers: Why do listening patterns differ on weekends?
        """
        logger.info("Building weekend vs weekday analysis...")

        comparison = df.groupBy('is_weekend').agg(
            count('*').alias('total_plays'),
            avg('valence').alias('avg_valence'),
            avg('energy').alias('avg_energy'),
            avg('danceability').alias('avg_danceability'),
            avg('tempo').alias('avg_tempo'),
            avg('acousticness').alias('avg_acousticness'),
            stddev('valence').alias('valence_variance'),
            stddev('energy').alias('energy_variance')
        ).withColumn('period', when(col('is_weekend'), 'Weekend').otherwise('Weekday')) \
          .withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
          .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
          .withColumn('avg_danceability', spark_round(col('avg_danceability'), 3)) \
          .withColumn('avg_tempo', spark_round(col('avg_tempo'), 1)) \
          .withColumn('avg_acousticness', spark_round(col('avg_acousticness'), 3)) \
          .withColumn('valence_variance', spark_round(col('valence_variance'), 3)) \
          .withColumn('energy_variance', spark_round(col('energy_variance'), 3)) \
          .select('period', 'total_plays', 'avg_valence', 'avg_energy',
                  'avg_danceability', 'avg_tempo', 'avg_acousticness',
                  'valence_variance', 'energy_variance') \
          .orderBy(desc('is_weekend'))

        logger.info(f"✅ Created weekend vs weekday analysis")
        return comparison

    def build_mood_shift_analysis(self, df: DataFrame) -> DataFrame:
        """
        DIAGNOSTIC: Identify when mood shifts occur (valence/energy changes).

        Answers: Why does mood shift at certain times?
        """
        logger.info("Building mood shift analysis...")

        # Order by time and calculate mood changes
        window = Window.orderBy('played_at')

        shifts = df.withColumn('prev_valence', lag('valence', 1).over(window)) \
            .withColumn('prev_energy', lag('energy', 1).over(window)) \
            .withColumn('valence_change', col('valence') - col('prev_valence')) \
            .withColumn('energy_change', col('energy') - col('prev_energy')) \
            .filter(col('prev_valence').isNotNull())

        # Categorize shift magnitude
        shift_summary = shifts.withColumn(
            'shift_magnitude',
            when((col('valence_change').between(-0.2, 0.2)) &
                 (col('energy_change').between(-0.2, 0.2)), 'Stable')
            .when((spark_abs(col('valence_change')) > 0.5) |
                  (spark_abs(col('energy_change')) > 0.5), 'Major_Shift')
            .otherwise('Minor_Shift')
        )

        # Aggregate by hour and shift type
        shift_patterns = shift_summary.groupBy('hour_of_day', 'shift_magnitude').agg(
            count('*').alias('shift_count'),
            avg('valence_change').alias('avg_valence_change'),
            avg('energy_change').alias('avg_energy_change')
        ).withColumn('avg_valence_change', spark_round(col('avg_valence_change'), 3)) \
          .withColumn('avg_energy_change', spark_round(col('avg_energy_change'), 3)) \
          .orderBy('hour_of_day', desc('shift_count'))

        logger.info(f"✅ Created mood shift analysis")
        return shift_patterns

    def build_part_of_day_mood_drivers(self, df: DataFrame) -> DataFrame:
        """
        DIAGNOSTIC: Identify which audio features drive mood in different parts of day.

        Answers: Why are mornings more energetic? Why are evenings more acoustic?
        """
        logger.info("Building part-of-day mood drivers...")

        drivers = df.groupBy('part_of_day').agg(
            count('*').alias('total_plays'),
            avg('valence').alias('avg_valence'),
            avg('energy').alias('avg_energy'),
            avg('danceability').alias('avg_danceability'),
            avg('acousticness').alias('avg_acousticness'),
            avg('instrumentalness').alias('avg_instrumentalness'),
            avg('tempo').alias('avg_tempo'),
            stddev('valence').alias('valence_consistency'),
            stddev('energy').alias('energy_consistency')
        ).withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
          .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
          .withColumn('avg_danceability', spark_round(col('avg_danceability'), 3)) \
          .withColumn('avg_acousticness', spark_round(col('avg_acousticness'), 3)) \
          .withColumn('avg_instrumentalness', spark_round(col('avg_instrumentalness'), 3)) \
          .withColumn('avg_tempo', spark_round(col('avg_tempo'), 1)) \
          .withColumn('valence_consistency', spark_round(col('valence_consistency'), 3)) \
          .withColumn('energy_consistency', spark_round(col('energy_consistency'), 3)) \
          .orderBy(
              when(col('part_of_day') == 'morning', 1)
              .when(col('part_of_day') == 'afternoon', 2)
              .when(col('part_of_day') == 'evening', 3)
              .otherwise(4)
          )

        logger.info(f"✅ Created part-of-day mood drivers")
        return drivers

    def write_gold_table(self, df: DataFrame, table_name: str, gold_path: str):
        """Write Gold layer table to Delta Lake."""
        output_path = f"{gold_path}/diagnostic/{table_name}"
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
        """Build all diagnostic analytics tables."""
        logger.info("=" * 80)
        logger.info("BUILDING DIAGNOSTIC ANALYTICS - GOLD LAYER")
        logger.info("=" * 80)

        # Load Silver data
        silver_df = self.load_silver_data(silver_path)

        # Build all diagnostic tables
        tables = {
            'mood_time_correlations': self.build_mood_time_correlations(silver_df),
            'feature_correlations': self.build_feature_correlations(silver_df),
            'weekend_vs_weekday': self.build_weekend_vs_weekday_analysis(silver_df),
            'mood_shift_patterns': self.build_mood_shift_analysis(silver_df),
            'part_of_day_drivers': self.build_part_of_day_mood_drivers(silver_df),
        }

        # Write all tables
        for table_name, table_df in tables.items():
            self.write_gold_table(table_df, table_name, gold_path)

        logger.info("=" * 80)
        logger.info("✅ DIAGNOSTIC ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Created {len(tables)} analytical tables:")
        for table_name in tables.keys():
            logger.info(f"  - {table_name}")


def main():
    """Main execution."""
    logger.info("Starting Diagnostic Analytics build...")

    # Paths
    silver_path = '/app/data/silver/listening_with_features'
    gold_path = '/app/data/gold'

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("DiagnosticAnalytics") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        analytics = DiagnosticAnalytics(spark)
        analytics.build_all(silver_path, gold_path)

        logger.info("=" * 80)
        logger.info("✅ Diagnostic Analytics build SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Diagnostic Analytics build FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
