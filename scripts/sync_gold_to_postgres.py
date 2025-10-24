"""
Sync Gold Layer Delta Tables to PostgreSQL for Superset

This script reads all Gold layer Delta tables and syncs them to PostgreSQL,
making them accessible to Apache Superset for visualization.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from utils.logger import setup_logger

logger = setup_logger('sync_gold_to_postgres')


def sync_table_to_postgres(spark: SparkSession, table_path: str, table_name: str, jdbc_url: str):
    """Sync a single Delta table to PostgreSQL."""
    try:
        # Read Delta table
        df = spark.read.format('delta').load(table_path)
        record_count = df.count()

        logger.info(f"Syncing {table_name}: {record_count} records")

        # Write to PostgreSQL
        df.write \
            .format('jdbc') \
            .option('url', jdbc_url) \
            .option('dbtable', f'public.{table_name}') \
            .option('user', 'superset') \
            .option('password', 'superset') \
            .option('driver', 'org.postgresql.Driver') \
            .mode('overwrite') \
            .save()

        logger.info(f"✅ Synced {table_name}: {record_count} records")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to sync {table_name}: {e}")
        return False


def sync_all_gold_tables(spark: SparkSession):
    """Sync all Gold layer tables to PostgreSQL."""

    jdbc_url = 'jdbc:postgresql://postgres:5432/superset'

    # Define all Gold tables
    tables = {
        # Descriptive Analytics
        'listening_patterns_by_time': '/app/data/gold/descriptive/listening_patterns_by_time',
        'top_tracks_by_mood': '/app/data/gold/descriptive/top_tracks_by_mood',
        'temporal_trends': '/app/data/gold/descriptive/temporal_trends',
        'audio_feature_distributions': '/app/data/gold/descriptive/audio_feature_distributions',
        'feature_source_coverage': '/app/data/gold/descriptive/feature_source_coverage',

        # Diagnostic Analytics
        'mood_time_correlations': '/app/data/gold/diagnostic/mood_time_correlations',
        'feature_correlations': '/app/data/gold/diagnostic/feature_correlations',
        'weekend_vs_weekday': '/app/data/gold/diagnostic/weekend_vs_weekday',
        'mood_shift_patterns': '/app/data/gold/diagnostic/mood_shift_patterns',
        'part_of_day_drivers': '/app/data/gold/diagnostic/part_of_day_drivers',

        # Predictive Analytics
        'mood_predictions': '/app/data/gold/predictive/mood_predictions',
        'energy_forecasts': '/app/data/gold/predictive/energy_forecasts',
        'mood_classifications': '/app/data/gold/predictive/mood_classifications',
        'model_performance_metrics': '/app/data/gold/predictive/model_performance_metrics',

        # Prescriptive Analytics
        'mood_improvement_recommendations': '/app/data/gold/prescriptive/mood_improvement_recommendations',
        'optimal_listening_times': '/app/data/gold/prescriptive/optimal_listening_times',
        'personalized_playlist_suggestions': '/app/data/gold/prescriptive/personalized_playlist_suggestions',
        'mood_intervention_triggers': '/app/data/gold/prescriptive/mood_intervention_triggers',

        # Cognitive Analytics
        'mood_state_clusters': '/app/data/gold/cognitive/mood_state_clusters',
        'listening_anomalies': '/app/data/gold/cognitive/listening_anomalies',
        'sequential_patterns': '/app/data/gold/cognitive/sequential_patterns',
        'behavioral_segments': '/app/data/gold/cognitive/behavioral_segments',
    }

    logger.info("=" * 80)
    logger.info("SYNCING GOLD LAYER TO POSTGRESQL FOR SUPERSET")
    logger.info("=" * 80)
    logger.info(f"Target database: {jdbc_url}")
    logger.info(f"Total tables to sync: {len(tables)}")
    logger.info("=" * 80)

    success_count = 0
    failed_count = 0

    for table_name, table_path in tables.items():
        if sync_table_to_postgres(spark, table_path, table_name, jdbc_url):
            success_count += 1
        else:
            failed_count += 1

    logger.info("=" * 80)
    logger.info("SYNC SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Successfully synced: {success_count} tables")
    logger.info(f"❌ Failed: {failed_count} tables")
    logger.info("=" * 80)

    if success_count > 0:
        logger.info("")
        logger.info("🎉 Next Steps:")
        logger.info("1. Access Superset at http://localhost:8088")
        logger.info("2. Login with admin/admin")
        logger.info("3. Go to Settings → Database Connections")
        logger.info("4. Add PostgreSQL connection:")
        logger.info("   - SQLAlchemy URI: postgresql://superset:superset@postgres:5432/superset")
        logger.info("5. Go to Data → Datasets and sync tables")
        logger.info("6. Start creating charts and dashboards!")


def main():
    """Main execution."""
    logger.info("Starting PostgreSQL sync...")

    # Initialize Spark with Delta Lake and PostgreSQL support
    spark = SparkSession.builder \
        .appName("SyncGoldToPostgres") \
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.2.1,org.postgresql:postgresql:42.6.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        sync_all_gold_tables(spark)

        logger.info("=" * 80)
        logger.info("✅ PostgreSQL sync SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ PostgreSQL sync FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
