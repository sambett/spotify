"""
Data Quality Validation Framework

Provides automated data quality checks for the Spotify Analytics pipeline.
These tests ensure data integrity at each layer (Bronze, Silver, Gold).

Academic Note:
These validations demonstrate proper data quality engineering practices,
even though the underlying audio features are synthetic.
"""
from typing import Dict, List, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, isnan, when, min, max, avg, stddev
from utils.logger import setup_logger

logger = setup_logger('data_quality')


class DataQualityValidator:
    """Validates data quality across pipeline layers."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.validation_results = []

    def check_null_values(self, df: DataFrame, column_name: str,
                          max_null_percentage: float = 5.0) -> Tuple[bool, str]:
        """
        Check if null percentage in a column exceeds threshold.

        Args:
            df: DataFrame to check
            column_name: Column to validate
            max_null_percentage: Maximum acceptable null percentage

        Returns:
            (passed, message)
        """
        total_count = df.count()
        null_count = df.filter(col(column_name).isNull() | isnan(col(column_name))).count()
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0

        passed = null_percentage <= max_null_percentage

        message = (
            f"{'✅ PASS' if passed else '❌ FAIL'}: Column '{column_name}' has "
            f"{null_percentage:.2f}% nulls (threshold: {max_null_percentage}%)"
        )

        return passed, message

    def check_value_range(self, df: DataFrame, column_name: str,
                         min_value: float, max_value: float) -> Tuple[bool, str]:
        """
        Check if all values in a column are within expected range.

        Args:
            df: DataFrame to check
            column_name: Column to validate
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value

        Returns:
            (passed, message)
        """
        out_of_range = df.filter(
            (col(column_name) < min_value) | (col(column_name) > max_value)
        ).count()

        passed = out_of_range == 0

        if passed:
            message = f"✅ PASS: All '{column_name}' values are within [{min_value}, {max_value}]"
        else:
            message = f"❌ FAIL: {out_of_range} rows have '{column_name}' outside [{min_value}, {max_value}]"

        return passed, message

    def check_row_count(self, df: DataFrame, min_rows: int,
                        layer_name: str = "table") -> Tuple[bool, str]:
        """
        Check if DataFrame has minimum number of rows.

        Args:
            df: DataFrame to check
            min_rows: Minimum acceptable row count
            layer_name: Name of the layer/table

        Returns:
            (passed, message)
        """
        row_count = df.count()
        passed = row_count >= min_rows

        message = (
            f"{'✅ PASS' if passed else '❌ FAIL'}: {layer_name} has "
            f"{row_count} rows (minimum: {min_rows})"
        )

        return passed, message

    def check_duplicates(self, df: DataFrame, key_columns: List[str],
                         layer_name: str = "table") -> Tuple[bool, str]:
        """
        Check for duplicate rows based on key columns.

        Args:
            df: DataFrame to check
            key_columns: Columns that define uniqueness
            layer_name: Name of the layer/table

        Returns:
            (passed, message)
        """
        total_count = df.count()
        distinct_count = df.select(*key_columns).distinct().count()
        duplicate_count = total_count - distinct_count

        passed = duplicate_count == 0

        if passed:
            message = f"✅ PASS: No duplicates in {layer_name}"
        else:
            message = f"⚠️  WARNING: {duplicate_count} duplicate rows in {layer_name}"

        return passed, message

    def check_schema_compliance(self, df: DataFrame, expected_columns: List[str],
                                layer_name: str = "table") -> Tuple[bool, str]:
        """
        Check if DataFrame has all expected columns.

        Args:
            df: DataFrame to check
            expected_columns: List of required column names
            layer_name: Name of the layer/table

        Returns:
            (passed, message)
        """
        actual_columns = set(df.columns)
        expected = set(expected_columns)
        missing = expected - actual_columns

        passed = len(missing) == 0

        if passed:
            message = f"✅ PASS: {layer_name} has all required columns"
        else:
            message = f"❌ FAIL: {layer_name} missing columns: {missing}"

        return passed, message

    def check_distribution(self, df: DataFrame, column_name: str,
                           expected_mean: float, tolerance: float = 0.3) -> Tuple[bool, str]:
        """
        Check if column mean is within expected range (for synthetic data validation).

        Args:
            df: DataFrame to check
            column_name: Column to validate
            expected_mean: Expected mean value
            tolerance: Acceptable deviation from mean

        Returns:
            (passed, message)
        """
        stats = df.select(avg(col(column_name))).collect()[0][0]
        actual_mean = float(stats) if stats is not None else 0.0

        deviation = abs(actual_mean - expected_mean)
        passed = deviation <= tolerance

        message = (
            f"{'✅ PASS' if passed else '⚠️  WARNING'}: '{column_name}' mean is "
            f"{actual_mean:.3f} (expected: {expected_mean:.3f} ± {tolerance:.3f})"
        )

        return passed, message

    def validate_bronze_layer(self, bronze_path: str) -> Dict[str, List[Tuple[bool, str]]]:
        """
        Validate Bronze layer data quality.

        Returns:
            Dictionary of validation results by table
        """
        logger.info("=" * 80)
        logger.info("VALIDATING BRONZE LAYER DATA QUALITY")
        logger.info("=" * 80)

        results = {}

        # Validate Kaggle tracks
        try:
            logger.info("\n📊 Kaggle Tracks Bronze:")
            kaggle_df = self.spark.read.format('delta').load(f"{bronze_path}/kaggle_tracks_bronze")

            kaggle_checks = [
                self.check_row_count(kaggle_df, 100000, "kaggle_tracks_bronze"),
                self.check_schema_compliance(kaggle_df, [
                    'track_id', 'name', 'artists', 'valence', 'energy',
                    'danceability', 'acousticness', 'tempo'
                ], "kaggle_tracks_bronze"),
                self.check_value_range(kaggle_df, 'valence', 0.0, 1.0),
                self.check_value_range(kaggle_df, 'energy', 0.0, 1.0),
                self.check_value_range(kaggle_df, 'danceability', 0.0, 1.0),
                self.check_value_range(kaggle_df, 'tempo', 40.0, 250.0),
                self.check_null_values(kaggle_df, 'track_id', max_null_percentage=0.0),
            ]

            for passed, message in kaggle_checks:
                logger.info(f"  {message}")

            results['kaggle_tracks_bronze'] = kaggle_checks

        except Exception as e:
            logger.error(f"  ❌ Failed to validate kaggle_tracks_bronze: {e}")
            results['kaggle_tracks_bronze'] = [(False, f"Validation error: {e}")]

        # Validate Spotify listening history
        try:
            logger.info("\n📊 Spotify Listening History Bronze:")
            spotify_df = self.spark.read.format('delta').load(f"{bronze_path}/spotify_listening_bronze")

            spotify_checks = [
                self.check_row_count(spotify_df, 100, "spotify_listening_bronze"),
                self.check_schema_compliance(spotify_df, [
                    'played_at', 'track_id', 'track_name', 'artist_name'
                ], "spotify_listening_bronze"),
                self.check_null_values(spotify_df, 'played_at', max_null_percentage=0.0),
                self.check_duplicates(spotify_df, ['played_at', 'track_id'], "spotify_listening_bronze"),
            ]

            for passed, message in spotify_checks:
                logger.info(f"  {message}")

            results['spotify_listening_bronze'] = spotify_checks

        except Exception as e:
            logger.error(f"  ❌ Failed to validate spotify_listening_bronze: {e}")
            results['spotify_listening_bronze'] = [(False, f"Validation error: {e}")]

        return results

    def validate_silver_layer(self, silver_path: str) -> Dict[str, List[Tuple[bool, str]]]:
        """
        Validate Silver layer data quality.

        Returns:
            Dictionary of validation results
        """
        logger.info("=" * 80)
        logger.info("VALIDATING SILVER LAYER DATA QUALITY")
        logger.info("=" * 80)

        results = {}

        try:
            logger.info("\n📊 Silver Listening with Features:")
            silver_df = self.spark.read.format('delta').load(silver_path)

            silver_checks = [
                self.check_row_count(silver_df, 100, "listening_with_features"),
                self.check_schema_compliance(silver_df, [
                    'played_at', 'track_name', 'artist_name', 'valence', 'energy',
                    'danceability', 'hour_of_day', 'day_of_week'
                ], "listening_with_features"),
                self.check_value_range(silver_df, 'valence', 0.0, 1.0),
                self.check_value_range(silver_df, 'energy', 0.0, 1.0),
                self.check_value_range(silver_df, 'hour_of_day', 0, 23),
                self.check_value_range(silver_df, 'day_of_week', 0, 6),
                self.check_null_values(silver_df, 'valence', max_null_percentage=1.0),
            ]

            # Check synthetic feature distributions (should be uniform-ish)
            silver_checks.extend([
                self.check_distribution(silver_df, 'valence', 0.5, tolerance=0.25),
                self.check_distribution(silver_df, 'energy', 0.525, tolerance=0.25),
            ])

            for passed, message in silver_checks:
                logger.info(f"  {message}")

            results['listening_with_features'] = silver_checks

        except Exception as e:
            logger.error(f"  ❌ Failed to validate Silver layer: {e}")
            results['listening_with_features'] = [(False, f"Validation error: {e}")]

        return results

    def validate_gold_layer(self, gold_path: str) -> Dict[str, List[Tuple[bool, str]]]:
        """
        Validate Gold layer data quality.

        Returns:
            Dictionary of validation results by analytics type
        """
        logger.info("=" * 80)
        logger.info("VALIDATING GOLD LAYER DATA QUALITY")
        logger.info("=" * 80)

        results = {}
        gold_tables = [
            ('descriptive/listening_summary', 'listening_summary', ['total_tracks_played', 'unique_tracks']),
            ('descriptive/hourly_patterns', 'hourly_patterns', ['hour_of_day', 'listen_count']),
            ('diagnostic/mood_correlations', 'mood_correlations', ['hour_of_day', 'avg_valence']),
            ('predictive/mood_predictions', 'mood_predictions', ['valence', 'predicted_valence']),
            ('cognitive/mood_state_clusters', 'mood_state_clusters', ['cluster', 'member_count']),
        ]

        for table_path, table_name, key_cols in gold_tables:
            try:
                logger.info(f"\n📊 {table_name}:")
                df = self.spark.read.format('delta').load(f"{gold_path}/{table_path}")

                checks = [
                    self.check_row_count(df, 1, table_name),
                    self.check_schema_compliance(df, key_cols, table_name),
                ]

                for passed, message in checks:
                    logger.info(f"  {message}")

                results[table_name] = checks

            except Exception as e:
                logger.warning(f"  ⚠️  Could not validate {table_name}: {e}")
                results[table_name] = [(False, f"Table not found or error: {e}")]

        return results

    def generate_validation_report(self, all_results: Dict[str, Dict[str, List[Tuple[bool, str]]]]) -> str:
        """
        Generate summary validation report.

        Args:
            all_results: Results from all layer validations

        Returns:
            Formatted report string
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("DATA QUALITY VALIDATION SUMMARY")
        logger.info("=" * 80)

        total_checks = 0
        passed_checks = 0
        failed_checks = 0

        for layer_name, layer_results in all_results.items():
            logger.info(f"\n{layer_name.upper()} LAYER:")
            for table_name, checks in layer_results.items():
                for passed, _ in checks:
                    total_checks += 1
                    if passed:
                        passed_checks += 1
                    else:
                        failed_checks += 1

        logger.info("")
        logger.info(f"Total Checks:  {total_checks}")
        logger.info(f"Passed:        {passed_checks} ({'%.1f' % (passed_checks/total_checks*100 if total_checks > 0 else 0)}%)")
        logger.info(f"Failed:        {failed_checks} ({'%.1f' % (failed_checks/total_checks*100 if total_checks > 0 else 0)}%)")
        logger.info("=" * 80)

        if failed_checks == 0:
            logger.info("✅ ALL DATA QUALITY CHECKS PASSED")
        else:
            logger.warning(f"⚠️  {failed_checks} CHECK(S) FAILED - Review logs above")

        logger.info("=" * 80)

        return f"{passed_checks}/{total_checks} checks passed"


def validate_all_layers(spark: SparkSession, bronze_path: str = '/app/data/bronze',
                        silver_path: str = '/app/data/silver/listening_with_features',
                        gold_path: str = '/app/data/gold') -> str:
    """
    Run all data quality validations.

    Args:
        spark: SparkSession
        bronze_path: Path to Bronze layer
        silver_path: Path to Silver layer
        gold_path: Path to Gold layer

    Returns:
        Summary string
    """
    validator = DataQualityValidator(spark)

    all_results = {
        'bronze': validator.validate_bronze_layer(bronze_path),
        'silver': validator.validate_silver_layer(silver_path),
        'gold': validator.validate_gold_layer(gold_path),
    }

    summary = validator.generate_validation_report(all_results)

    return summary


if __name__ == '__main__':
    """Run data quality validation standalone."""
    from pyspark.sql import SparkSession

    logger.info("Starting Data Quality Validation...")

    spark = SparkSession.builder \
        .appName("DataQualityValidation") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        summary = validate_all_layers(spark)
        logger.info(f"\n✅ Validation complete: {summary}")

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        spark.stop()
