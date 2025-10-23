"""
Delta Lake writer for Bronze layer - Optimized for large datasets.
"""
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from utils import setup_logger


logger = setup_logger(__name__)


class DeltaWriter:
    """Writes DataFrames to Delta Lake Bronze layer with optimized execution."""
    
    @staticmethod
    def write_listening_history(df: DataFrame, path: Path, mode: str = 'append') -> None:
        """Write listening history to Delta Lake."""
        logger.info(f"Writing listening history to {path} (mode={mode})")

        # Add partition column
        df_with_partition = df.withColumn('date', F.to_date(F.col('played_at')))

        if mode == 'append' and path.exists():
            logger.info("Deduplicating against existing data...")
            existing_df = df.sparkSession.read.format('delta').load(str(path))
            existing_keys = existing_df.select('track_id', 'played_at').distinct()

            df_with_partition = df_with_partition.join(
                existing_keys,
                ['track_id', 'played_at'],
                'left_anti'
            )

        # Write without calling count() to avoid Spark worker crashes
        logger.info("Writing to Delta Lake...")
        (df_with_partition
            .write
            .format('delta')
            .mode(mode)
            .partitionBy('date')
            .save(str(path)))

        logger.info(f"✅ Listening history written to {path}")
    
    @staticmethod
    def write_tracks_features(df: DataFrame, path: Path, mode: str = 'append') -> None:
        """Write tracks features to Delta Lake."""
        logger.info(f"Writing tracks features to {path} (mode={mode})")
        
        if mode == 'append' and path.exists():
            logger.info("Merging with existing data...")
            from delta.tables import DeltaTable
            
            delta_table = DeltaTable.forPath(df.sparkSession, str(path))
            (delta_table.alias('target')
                .merge(df.alias('source'), 'target.track_id = source.track_id')
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute())
            logger.info(f"✅ Tracks features merged")
        else:
            df.write.format('delta').mode(mode).save(str(path))
            logger.info(f"✅ Tracks features written to {path}")
    
    @staticmethod
    def write_kaggle_tracks(df: DataFrame, path: Path, mode: str = 'overwrite') -> None:
        """Write Kaggle tracks to Delta Lake."""
        logger.info(f"Writing Kaggle tracks to {path} (mode={mode})")
        
        # Direct write with coalesce to reduce partitions for better performance
        (df.coalesce(10)
            .write
            .format('delta')
            .mode(mode)
            .option("overwriteSchema", "true")
            .save(str(path)))
        
        logger.info(f"✅ Kaggle tracks written to {path}")
