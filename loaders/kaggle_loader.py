"""
Kaggle dataset loader.
Loads and normalizes CSV data from Kaggle Spotify dataset.
"""
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from utils import setup_logger


logger = setup_logger(__name__)


class KaggleLoader:
    """
    Loader for Kaggle Spotify tracks dataset.
    
    Responsibilities:
    - Load CSV file with proper encoding and error handling
    - Handle missing or malformed data
    - Provide initial data quality logging
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize Kaggle loader.
        
        Args:
            spark: Spark session
        """
        self.spark = spark
    
    def load_csv(
        self, 
        csv_path: Path,
        header: bool = True,
        infer_schema: bool = True
    ) -> Optional[DataFrame]:
        """
        Load Kaggle CSV file.
        
        Args:
            csv_path: Path to CSV file
            header: Whether CSV has header row
            infer_schema: Whether to infer schema automatically
        
        Returns:
            DataFrame or None if loading fails
        """
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return None
        
        logger.info(f"📂 Loading Kaggle dataset from {csv_path}")
        
        try:
            df = self.spark.read \
                .option("header", str(header).lower()) \
                .option("inferSchema", str(infer_schema).lower()) \
                .option("mode", "PERMISSIVE") \
                .option("columnNameOfCorruptRecord", "_corrupt_record") \
                .csv(str(csv_path))
            
            # Log basic stats
            row_count = df.count()
            col_count = len(df.columns)
            
            logger.info(f"✅ Loaded {row_count:,} rows, {col_count} columns")
            
            # Check for corrupt records
            if "_corrupt_record" in df.columns:
                corrupt_count = df.filter(F.col("_corrupt_record").isNotNull()).count()
                if corrupt_count > 0:
                    logger.warning(f"⚠️  Found {corrupt_count} corrupt records")
                df = df.drop("_corrupt_record")
            
            # Log column names
            logger.debug(f"Columns: {', '.join(df.columns)}")
            
            # Check for nulls in critical columns
            null_counts = {}
            for col in df.columns:
                null_count = df.filter(F.col(col).isNull()).count()
                if null_count > 0:
                    null_counts[col] = null_count
            
            if null_counts:
                logger.debug("Null counts:")
                for col, count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    pct = (count / row_count) * 100
                    logger.debug(f"  {col}: {count:,} ({pct:.1f}%)")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}", exc_info=True)
            return None
    
    def validate_expected_columns(self, df: DataFrame, expected_cols: list) -> bool:
        """
        Validate that DataFrame has expected columns.
        
        Args:
            df: DataFrame to validate
            expected_cols: List of expected column names
        
        Returns:
            True if all expected columns present
        """
        actual_cols = set(df.columns)
        expected_set = set(expected_cols)
        
        missing = expected_set - actual_cols
        extra = actual_cols - expected_set
        
        if missing:
            logger.warning(f"Missing expected columns: {missing}")
        
        if extra:
            logger.debug(f"Extra columns (will be ignored): {extra}")
        
        return len(missing) == 0
