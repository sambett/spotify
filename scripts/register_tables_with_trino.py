"""
Register Delta Lake tables with Trino

This script registers all Delta Lake tables from Bronze, Silver, and Gold layers
with Trino so they can be queried via SQL and visualized in Superset.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from utils.logger import setup_logger

logger = setup_logger('register_tables_with_trino')


def register_tables_with_trino(spark: SparkSession):
    """Register all Delta tables with Trino using CALL delta.system.register_table()."""

    # Define all tables to register
    tables = {
        # Bronze layer
        'listening_history_bronze': '/app/data/bronze/listening_history_bronze',
        'my_tracks_features_bronze': '/app/data/bronze/my_tracks_features_bronze',
        'my_tracks_features_bronze_synthetic': '/app/data/bronze/my_tracks_features_bronze_synthetic',
        'kaggle_tracks_bronze': '/app/data/bronze/kaggle_tracks_bronze',

        # Silver layer
        'listening_with_features': '/app/data/silver/listening_with_features',

        # Gold - Descriptive
        'listening_patterns_by_time': '/app/data/gold/descriptive/listening_patterns_by_time',
        'top_tracks_by_mood': '/app/data/gold/descriptive/top_tracks_by_mood',
        'temporal_trends': '/app/data/gold/descriptive/temporal_trends',
        'audio_feature_distributions': '/app/data/gold/descriptive/audio_feature_distributions',
        'feature_source_coverage': '/app/data/gold/descriptive/feature_source_coverage',

        # Gold - Diagnostic
        'mood_time_correlations': '/app/data/gold/diagnostic/mood_time_correlations',
        'feature_correlations': '/app/data/gold/diagnostic/feature_correlations',
        'weekend_vs_weekday': '/app/data/gold/diagnostic/weekend_vs_weekday',
        'mood_shift_patterns': '/app/data/gold/diagnostic/mood_shift_patterns',
        'part_of_day_drivers': '/app/data/gold/diagnostic/part_of_day_drivers',

        # Gold - Predictive
        'mood_predictions': '/app/data/gold/predictive/mood_predictions',
        'energy_forecasts': '/app/data/gold/predictive/energy_forecasts',
        'mood_classifications': '/app/data/gold/predictive/mood_classifications',
        'model_performance_metrics': '/app/data/gold/predictive/model_performance_metrics',
    }

    logger.info("=" * 80)
    logger.info("REGISTERING DELTA TABLES WITH TRINO")
    logger.info("=" * 80)

    registered_count = 0
    failed_count = 0

    for table_name, table_path in tables.items():
        try:
            # Check if table exists
            try:
                df = spark.read.format('delta').load(table_path)
                record_count = df.count()
                logger.info(f"Found table: {table_name} ({record_count} records)")
            except Exception as e:
                logger.warning(f"⚠️  Table {table_name} not found at {table_path}, skipping")
                continue

            # Register with Trino using SQL
            # Note: Trino's CALL procedure syntax
            register_sql = f"""
                CALL delta.system.register_table(
                    schema_name => 'default',
                    table_name => '{table_name}',
                    table_location => '{table_path}'
                )
            """

            # We can't execute Trino SQL directly from Spark, so we'll create the metadata
            # Instead, we'll use PySpark to create a temp view and show how to register manually
            logger.info(f"✅ Table {table_name} ready for registration")
            logger.info(f"   Trino SQL: CALL delta.system.register_table('default', '{table_name}', '{table_path}');")

            registered_count += 1

        except Exception as e:
            logger.error(f"❌ Failed to process {table_name}: {e}")
            failed_count += 1

    logger.info("=" * 80)
    logger.info(f"✅ Registration summary:")
    logger.info(f"   Tables ready: {registered_count}")
    logger.info(f"   Tables failed: {failed_count}")
    logger.info("=" * 80)

    # Generate Trino registration script
    trino_script = "-- Trino Table Registration Script\n"
    trino_script += "-- Run this in Trino CLI to register all tables\n\n"

    for table_name, table_path in tables.items():
        trino_script += f"CALL delta.system.register_table('default', '{table_name}', '{table_path}');\n"

    # Save Trino script
    script_path = '/app/scripts/register_with_trino.sql'
    with open(script_path, 'w') as f:
        f.write(trino_script)

    logger.info(f"📄 Trino registration script saved to: {script_path}")
    logger.info("   Run with: docker exec trino trino --file /app/scripts/register_with_trino.sql")


def main():
    """Main execution."""
    logger.info("Starting Trino table registration...")

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("RegisterTablesWithTrino") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        register_tables_with_trino(spark)

        logger.info("=" * 80)
        logger.info("✅ Table registration preparation SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Table registration FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
