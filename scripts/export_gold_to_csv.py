"""
Export Gold Layer Tables to CSV for Superset

This script exports all Gold layer Delta tables to CSV format
so they can be easily uploaded to Apache Superset for visualization.
"""
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from utils.logger import setup_logger

logger = setup_logger('export_gold_to_csv')


def export_table_to_csv(spark: SparkSession, table_path: str, table_name: str, output_dir: str):
    """Export a Delta table to CSV."""
    try:
        # Read Delta table
        df = spark.read.format('delta').load(table_path)
        record_count = df.count()

        logger.info(f"Exporting {table_name}: {record_count} records")

        # Write to CSV
        output_path = f"{output_dir}/{table_name}.csv"
        df.coalesce(1) \
            .write \
            .mode('overwrite') \
            .option('header', 'true') \
            .csv(output_path)

        # Find the actual CSV file (Spark creates a directory)
        csv_files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
        if csv_files:
            # Move the CSV file to the parent directory with the correct name
            actual_csv = os.path.join(output_path, csv_files[0])
            final_csv = f"{output_dir}/{table_name}.csv.final"

            import shutil
            shutil.move(actual_csv, final_csv)
            shutil.rmtree(output_path)  # Remove the Spark output directory
            os.rename(final_csv, output_path.replace('.csv', '_final.csv'))

            logger.info(f"✅ Exported: {table_name}_final.csv")
        else:
            logger.warning(f"⚠️  No CSV file generated for {table_name}")

    except Exception as e:
        logger.error(f"❌ Failed to export {table_name}: {e}")


def export_all_gold_tables(spark: SparkSession):
    """Export all Gold layer tables to CSV."""

    output_dir = '/app/data/exports'
    os.makedirs(output_dir, exist_ok=True)

    # Define all Gold tables
    tables = {
        # Descriptive
        'listening_patterns_by_time': '/app/data/gold/descriptive/listening_patterns_by_time',
        'top_tracks_by_mood': '/app/data/gold/descriptive/top_tracks_by_mood',
        'temporal_trends': '/app/data/gold/descriptive/temporal_trends',
        'audio_feature_distributions': '/app/data/gold/descriptive/audio_feature_distributions',
        'feature_source_coverage': '/app/data/gold/descriptive/feature_source_coverage',

        # Diagnostic
        'mood_time_correlations': '/app/data/gold/diagnostic/mood_time_correlations',
        'feature_correlations': '/app/data/gold/diagnostic/feature_correlations',
        'weekend_vs_weekday': '/app/data/gold/diagnostic/weekend_vs_weekday',
        'mood_shift_patterns': '/app/data/gold/diagnostic/mood_shift_patterns',
        'part_of_day_drivers': '/app/data/gold/diagnostic/part_of_day_drivers',

        # Predictive
        'mood_predictions': '/app/data/gold/predictive/mood_predictions',
        'energy_forecasts': '/app/data/gold/predictive/energy_forecasts',
        'mood_classifications': '/app/data/gold/predictive/mood_classifications',
        'model_performance_metrics': '/app/data/gold/predictive/model_performance_metrics',
    }

    logger.info("=" * 80)
    logger.info("EXPORTING GOLD LAYER TABLES TO CSV")
    logger.info("=" * 80)

    exported_count = 0
    for table_name, table_path in tables.items():
        export_table_to_csv(spark, table_path, table_name, output_dir)
        exported_count += 1

    logger.info("=" * 80)
    logger.info(f"✅ Export complete: {exported_count} tables")
    logger.info(f"📁 CSV files saved to: {output_dir}")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("1. Access Superset at http://localhost:8088")
    logger.info("2. Go to Data → Upload a CSV")
    logger.info("3. Upload each CSV file")
    logger.info("4. Create dashboards!")


def main():
    """Main execution."""
    logger.info("Starting CSV export...")

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("ExportGoldToCSV") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        export_all_gold_tables(spark)

        logger.info("=" * 80)
        logger.info("✅ CSV export SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ CSV export FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
