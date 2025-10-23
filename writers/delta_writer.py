"""
Delta Lake writer for Bronze layer.
Handles writing DataFrames to Delta Lake with proper configuration.
"""
from pathlib import Path
from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from utils import setup_logger


logger = setup_logger(__name__)


class DeltaWriter:
    """
    Writes DataFrames to Delta Lake Bronze layer.
    
    Responsibilities:
    - Write to Delta Lake format
    - Handle deduplication
    - Partition data appropriately
    - Provide write modes (append, overwrite)
    - Log write metrics
    """
    
    @staticmethod
    def write_listening_history(
        df: DataFrame,
        path: Path,
        mode: str = 'append'
    ) -> None:
        """
        Write listening history to Delta Lake.
        
        Partitioned by date(played_at) for efficient querying.
        Deduplicates on (track_id, played_at) to handle re-runs.
        
        Args:
            df: DataFrame to write
            path: Target Delta Lake path
            mode: Write mode ('append' or 'overwrite')
        """
        logger.info(f"Writing listening history to {path} (mode={mode})")
        
        write_count = df.count()
        
        if write_count == 0:
            logger.warning("No data to write")
            return
        
        # Add partition column
        df_with_partition = df.withColumn('date', F.to_date(F.col('played_at')))
        
        if mode == 'append' and path.exists():
            # Deduplicate against existing data
            logger.info("Deduplicating against existing data...")
            
            # Read existing data
            existing_df = df.sparkSession.read.format('delta').load(str(path))
            
            # Get existing (track_id, played_at) pairs
            existing_keys = existing_df.select('track_id', 'played_at').distinct()
            
            # Anti-join to get only new records
            df_with_partition = df_with_partition.join(
                existing_keys,
                ['track_id', 'played_at'],
                'left_anti'
            )
            
            new_count = df_with_partition.count()
            logger.info(f"After deduplication: {new_count}/{write_count} new records")
            
            if new_count == 0:
                logger.info("✅ No new records to write (all duplicates)")
                return
        
        # Write to Delta
        (df_with_partition
            .write
            .format('delta')
            .mode(mode)
            .partitionBy('date')
            .save(str(path)))
        
        logger.info(f"✅ Wrote {df_with_partition.count():,} rows to {path}")
    
    @staticmethod
    def write_tracks_features(
        df: DataFrame,
        path: Path,
        mode: str = 'append'
    ) -> None:
        """
        Write tracks features to Delta Lake.
        
        Deduplicates on track_id.
        Uses 'merge' strategy: updates existing tracks, adds new ones.
        
        Args:
            df: DataFrame to write
            path: Target Delta Lake path
            mode: Write mode ('append' or 'overwrite')
        """
        logger.info(f"Writing tracks features to {path} (mode={mode})")
        
        write_count = df.count()
        
        if write_count == 0:
            logger.warning("No data to write")
            return
        
        if mode == 'append' and path.exists():
            # Use Delta merge to upsert
            logger.info("Merging with existing data...")
            
            from delta.tables import DeltaTable
            
            delta_table = DeltaTable.forPath(df.sparkSession, str(path))
            
            # Merge: update if track_id exists, insert if new
            (delta_table.alias('target')
                .merge(
                    df.alias('source'),
                    'target.track_id = source.track_id'
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute())
            
            logger.info(f"✅ Merged {write_count:,} tracks")
        else:
            # Initial write or overwrite
            df.write.format('delta').mode(mode).save(str(path))
            logger.info(f"✅ Wrote {write_count:,} rows to {path}")
    
    @staticmethod
    def write_kaggle_tracks(
        df: DataFrame,
        path: Path,
        mode: str = 'overwrite'
    ) -> None:
        """
        Write Kaggle tracks to Delta Lake.
        
        Typically uses 'overwrite' since Kaggle data is static.
        
        Args:
            df: DataFrame to write
            path: Target Delta Lake path
            mode: Write mode ('overwrite' or 'append')
        """
        logger.info(f"Writing Kaggle tracks to {path} (mode={mode})")
        
        write_count = df.count()
        
        if write_count == 0:
            logger.warning("No data to write")
            return
        
        # Simple write (Kaggle data is static)
        df.write.format('delta').mode(mode).save(str(path))
        
        logger.info(f"✅ Wrote {write_count:,} rows to {path}")
    
    @staticmethod
    def optimize_table(spark_session, path: Path) -> None:
        """
        Optimize Delta table (compaction + Z-ordering).
        
        Run periodically to improve query performance.
        
        Args:
            spark_session: Spark session
            path: Delta table path
        """
        if not path.exists():
            logger.warning(f"Table does not exist: {path}")
            return
        
        logger.info(f"Optimizing Delta table: {path}")
        
        try:
            from delta.tables import DeltaTable
            
            delta_table = DeltaTable.forPath(spark_session, str(path))
            delta_table.optimize().executeCompaction()
            
            logger.info("✅ Optimization complete")
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
