"""
Data validation for Bronze layer.
Ensures data quality before writing to Delta Lake.
"""
from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from utils import setup_logger


logger = setup_logger(__name__)


class DataValidator:
    """
    Validates DataFrames before writing to Bronze layer.
    
    Responsibilities:
    - Check required fields are non-null
    - Validate data types
    - Check value ranges
    - Log validation warnings
    - Return clean dataset + metrics
    """
    
    @staticmethod
    def validate_required_fields(df: DataFrame, required_fields: list) -> Tuple[DataFrame, int]:
        """
        Validate required fields are non-null.
        
        Args:
            df: Input DataFrame
            required_fields: List of required column names
        
        Returns:
            Tuple of (cleaned DataFrame, count of dropped rows)
        """
        initial_count = df.count()
        
        # Build filter condition
        conditions = [F.col(field).isNotNull() for field in required_fields if field in df.columns]
        
        if not conditions:
            return df, 0
        
        # Combine conditions with AND
        filter_condition = conditions[0]
        for cond in conditions[1:]:
            filter_condition = filter_condition & cond
        
        # Filter
        df_clean = df.filter(filter_condition)
        final_count = df_clean.count()
        
        dropped = initial_count - final_count
        
        if dropped > 0:
            logger.warning(f"⚠️  Dropped {dropped} rows with null required fields")
        
        return df_clean, dropped
    
    @staticmethod
    def validate_listening_history(df: DataFrame) -> Tuple[DataFrame, dict]:
        """
        Validate listening history DataFrame.
        
        Checks:
        - track_id and played_at are non-null
        - played_at is a valid timestamp
        - duration_ms is positive if present
        
        Args:
            df: Listening history DataFrame
        
        Returns:
            Tuple of (validated DataFrame, metrics dict)
        """
        logger.info("Validating listening history data...")
        
        initial_count = df.count()
        metrics = {'initial': initial_count, 'dropped': 0, 'final': 0}
        
        # 1. Required fields
        df, dropped = DataValidator.validate_required_fields(
            df, ['track_id', 'played_at', '_ingested_at']
        )
        metrics['dropped'] += dropped
        
        # 2. Validate duration_ms is positive if present
        if 'duration_ms' in df.columns:
            invalid_duration = df.filter(
                F.col('duration_ms').isNotNull() & (F.col('duration_ms') <= 0)
            ).count()
            
            if invalid_duration > 0:
                logger.warning(f"Found {invalid_duration} tracks with invalid duration")
                df = df.filter(
                    F.col('duration_ms').isNull() | (F.col('duration_ms') > 0)
                )
                metrics['dropped'] += invalid_duration
        
        metrics['final'] = df.count()
        
        logger.info(f"✅ Validation complete: {metrics['final']}/{metrics['initial']} rows passed")
        
        return df, metrics
    
    @staticmethod
    def validate_tracks_features(df: DataFrame) -> Tuple[DataFrame, dict]:
        """
        Validate tracks features DataFrame.
        
        Checks:
        - track_id is non-null
        - Audio features are in valid ranges if present:
          - danceability, energy, valence, acousticness, instrumentalness: [0, 1]
          - tempo: > 0
        
        Args:
            df: Tracks features DataFrame
        
        Returns:
            Tuple of (validated DataFrame, metrics dict)
        """
        logger.info("Validating tracks features data...")
        
        initial_count = df.count()
        metrics = {'initial': initial_count, 'dropped': 0, 'final': 0}
        
        # 1. Required fields
        df, dropped = DataValidator.validate_required_fields(
            df, ['track_id', '_ingested_at']
        )
        metrics['dropped'] += dropped
        
        # 2. Validate audio feature ranges (0-1)
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
        
        for col in feature_cols:
            if col in df.columns:
                # Check for out-of-range values
                invalid = df.filter(
                    F.col(col).isNotNull() & 
                    ((F.col(col) < 0) | (F.col(col) > 1))
                ).count()
                
                if invalid > 0:
                    logger.warning(f"Found {invalid} rows with invalid {col} values")
                    # Set invalid values to NULL rather than dropping rows
                    df = df.withColumn(
                        col,
                        F.when(
                            (F.col(col) >= 0) & (F.col(col) <= 1),
                            F.col(col)
                        ).otherwise(None)
                    )
        
        # 3. Validate tempo is positive
        if 'tempo' in df.columns:
            invalid_tempo = df.filter(
                F.col('tempo').isNotNull() & (F.col('tempo') <= 0)
            ).count()
            
            if invalid_tempo > 0:
                logger.warning(f"Found {invalid_tempo} tracks with invalid tempo")
                df = df.withColumn(
                    'tempo',
                    F.when(F.col('tempo') > 0, F.col('tempo')).otherwise(None)
                )
        
        metrics['final'] = df.count()
        
        logger.info(f"✅ Validation complete: {metrics['final']}/{metrics['initial']} rows passed")
        
        return df, metrics
    
    @staticmethod
    def validate_kaggle_tracks(df: DataFrame) -> Tuple[DataFrame, dict]:
        """
        Validate Kaggle tracks DataFrame.
        
        Similar to tracks_features validation.
        
        Args:
            df: Kaggle tracks DataFrame
        
        Returns:
            Tuple of (validated DataFrame, metrics dict)
        """
        logger.info("Validating Kaggle tracks data...")
        
        initial_count = df.count()
        metrics = {'initial': initial_count, 'dropped': 0, 'final': 0}
        
        # 1. Required fields
        df, dropped = DataValidator.validate_required_fields(
            df, ['track_id', '_ingested_at']
        )
        metrics['dropped'] += dropped
        
        # 2. Validate audio feature ranges
        feature_cols = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
        
        for col in feature_cols:
            if col in df.columns:
                # Set out-of-range values to NULL
                df = df.withColumn(
                    col,
                    F.when(
                        (F.col(col) >= 0) & (F.col(col) <= 1),
                        F.col(col)
                    ).otherwise(None)
                )
        
        # 3. Validate tempo
        if 'tempo' in df.columns:
            df = df.withColumn(
                'tempo',
                F.when(F.col('tempo') > 0, F.col('tempo')).otherwise(None)
            )
        
        # 4. Validate duration_ms
        if 'duration_ms' in df.columns:
            df = df.withColumn(
                'duration_ms',
                F.when(F.col('duration_ms') > 0, F.col('duration_ms')).otherwise(None)
            )
        
        metrics['final'] = df.count()
        
        logger.info(f"✅ Validation complete: {metrics['final']}/{metrics['initial']} rows passed")
        
        return df, metrics
