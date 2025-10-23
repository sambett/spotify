"""
Kaggle dataset mapper.
Normalizes Kaggle CSV data to Bronze schema format.
"""
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, IntegerType, FloatType, BooleanType, TimestampType

from schemas import get_kaggle_tracks_schema
from utils import setup_logger


logger = setup_logger(__name__)


class KaggleMapper:
    """
    Maps Kaggle dataset to kaggle_tracks_bronze schema.
    
    Responsibilities:
    - Rename columns to match schema
    - Cast types correctly
    - Handle missing values
    - Add ingestion timestamp
    - Generate stable track IDs if missing
    """
    
    # Column name mappings from Kaggle CSV to Bronze schema
    COLUMN_MAPPINGS = {
        'track_id': 'track_id',
        'artists': 'artists',
        'album_name': 'album_name',
        'track_name': 'track_name',
        'popularity': 'popularity',
        'duration_ms': 'duration_ms',
        'explicit': 'explicit',
        'danceability': 'danceability',
        'energy': 'energy',
        'valence': 'valence',
        'acousticness': 'acousticness',
        'instrumentalness': 'instrumentalness',
        'tempo': 'tempo',
        'track_genre': 'track_genre'
    }
    
    @staticmethod
    def normalize_column_names(df: DataFrame) -> DataFrame:
        """
        Normalize column names to match Bronze schema.
        
        Handles common variations in Kaggle dataset column names.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with normalized column names
        """
        # Create lowercase mapping for case-insensitive matching
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Direct matches
            if col_lower in KaggleMapper.COLUMN_MAPPINGS:
                col_map[col] = KaggleMapper.COLUMN_MAPPINGS[col_lower]
            # Handle common variations
            elif col_lower in ['artist', 'artist_name']:
                col_map[col] = 'artists'
            elif col_lower in ['name', 'song_name']:
                col_map[col] = 'track_name'
            elif col_lower in ['album']:
                col_map[col] = 'album_name'
            elif col_lower in ['duration', 'duration_in_ms']:
                col_map[col] = 'duration_ms'
            elif col_lower in ['genre', 'genres']:
                col_map[col] = 'track_genre'
        
        # Rename columns
        for old_col, new_col in col_map.items():
            df = df.withColumnRenamed(old_col, new_col)
        
        logger.debug(f"Renamed {len(col_map)} columns")
        return df
    
    @staticmethod
    def cast_to_schema_types(df: DataFrame) -> DataFrame:
        """
        Cast columns to match Bronze schema types.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with correct types
        """
        # Define expected types
        type_casts = {
            'popularity': IntegerType(),
            'duration_ms': LongType(),
            'explicit': BooleanType(),
            'danceability': FloatType(),
            'energy': FloatType(),
            'valence': FloatType(),
            'acousticness': FloatType(),
            'instrumentalness': FloatType(),
            'tempo': FloatType()
        }
        
        for col, target_type in type_casts.items():
            if col in df.columns:
                try:
                    df = df.withColumn(col, F.col(col).cast(target_type))
                except Exception as e:
                    logger.warning(f"Failed to cast {col} to {target_type}: {e}")
        
        return df
    
    @staticmethod
    def generate_track_ids(df: DataFrame) -> DataFrame:
        """
        Generate stable track IDs if missing.

        Uses hash of track_name + artists + album_name for stability.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with track_id column
        """
        if 'track_id' not in df.columns:
            logger.info("track_id column missing, generating from track metadata...")

            df = df.withColumn(
                'track_id',
                F.md5(
                    F.concat_ws(
                        '|',
                        F.coalesce(F.col('track_name'), F.lit('')),
                        F.coalesce(F.col('artists'), F.lit('')),
                        F.coalesce(F.col('album_name'), F.lit(''))
                    )
                )
            )
        else:
            # Fill NULL track_ids if any exist (avoid count() to prevent crashes)
            logger.info("Generating track IDs for any NULL values...")
            df = df.withColumn(
                'track_id',
                F.when(
                    F.col('track_id').isNull(),
                    F.md5(
                        F.concat_ws(
                            '|',
                            F.coalesce(F.col('track_name'), F.lit('')),
                            F.coalesce(F.col('artists'), F.lit('')),
                            F.coalesce(F.col('album_name'), F.lit(''))
                        )
                    )
                ).otherwise(F.col('track_id'))
            )

        return df
    
    @staticmethod
    def add_ingestion_timestamp(df: DataFrame) -> DataFrame:
        """
        Add _ingested_at timestamp.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with _ingested_at column
        """
        return df.withColumn('_ingested_at', F.lit(datetime.now()).cast(TimestampType()))
    
    def map_to_bronze_schema(self, df: DataFrame) -> DataFrame:
        """
        Map Kaggle DataFrame to kaggle_tracks_bronze schema.

        Complete transformation pipeline:
        1. Normalize column names
        2. Generate track IDs if missing
        3. Cast to correct types
        4. Select only schema columns
        5. Add ingestion timestamp

        Args:
            df: Input DataFrame from Kaggle CSV

        Returns:
            DataFrame conforming to kaggle_tracks_bronze schema
        """
        logger.info("Mapping Kaggle data to Bronze schema...")

        # 1. Normalize column names
        df = self.normalize_column_names(df)

        # 2. Generate track IDs if needed
        df = self.generate_track_ids(df)

        # 3. Cast to correct types
        df = self.cast_to_schema_types(df)

        # 4. Get target schema fields
        schema = get_kaggle_tracks_schema()
        target_fields = [field.name for field in schema.fields if field.name != '_ingested_at']

        # Select available fields, adding NULL for missing ones
        select_exprs = []
        for field in target_fields:
            if field in df.columns:
                select_exprs.append(F.col(field))
            else:
                logger.debug(f"Adding NULL column for missing field: {field}")
                select_exprs.append(F.lit(None).alias(field))

        df = df.select(select_exprs)

        # 5. Add ingestion timestamp
        df = self.add_ingestion_timestamp(df)

        logger.info("✅ Mapped Kaggle data to Bronze schema")

        return df
