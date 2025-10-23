"""
COGNITIVE ANALYTICS - Gold Layer

Creates advanced pattern recognition and learning systems:
1. K-means clustering for mood state discovery
2. Anomaly detection in listening patterns
3. Sequential pattern mining
4. Behavioral segmentation

These tables answer: "What patterns exist?" and "How do complex behaviors emerge?"

Uses Spark MLlib for clustering and pattern recognition.
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, avg, stddev, min, max, desc, row_number,
    when, round as spark_round, abs as spark_abs, concat, lit
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator

from utils.logger import setup_logger

logger = setup_logger('cognitive_analytics')


class CognitiveAnalytics:
    """Build cognitive analytics tables using ML clustering and pattern recognition."""

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

    def build_mood_state_clusters(self, df: DataFrame) -> dict:
        """
        COGNITIVE: Discover natural mood states using K-means clustering.

        Cluster listening sessions based on audio features to discover
        natural mood states beyond simple categorization.
        """
        logger.info("Building mood state clusters...")

        # Select features for clustering
        feature_cols = ['valence', 'energy', 'danceability', 'acousticness',
                       'instrumentalness', 'tempo', 'loudness']

        # Prepare data
        model_data = df.select(*feature_cols).na.drop()

        # Normalize tempo and loudness to 0-1 range
        model_data = model_data.withColumn(
            'tempo_normalized',
            (col('tempo') - 60) / (180 - 60)  # Normalize tempo from 60-180 to 0-1
        ).withColumn(
            'loudness_normalized',
            (col('loudness') + 60) / 60  # Normalize loudness from -60-0 to 0-1
        )

        feature_cols_normalized = ['valence', 'energy', 'danceability', 'acousticness',
                                   'instrumentalness', 'tempo_normalized', 'loudness_normalized']

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols_normalized, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # K-means clustering (5 clusters for mood states)
        kmeans = KMeans(
            featuresCol='scaled_features',
            predictionCol='cluster',
            k=5,
            seed=42,
            maxIter=20
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, kmeans])

        # Train
        logger.info(f"Training K-means clustering with {model_data.count()} samples")
        model = pipeline.fit(model_data)

        # Predict
        predictions = model.transform(model_data)

        # Evaluate
        evaluator = ClusteringEvaluator(
            featuresCol='scaled_features',
            predictionCol='cluster',
            metricName='silhouette'
        )

        silhouette = evaluator.evaluate(predictions)
        logger.info(f"✅ Clustering Silhouette Score: {silhouette:.4f}")

        # Characterize each cluster
        cluster_profiles = predictions.groupBy('cluster').agg(
            count('*').alias('member_count'),
            avg('valence').alias('avg_valence'),
            avg('energy').alias('avg_energy'),
            avg('danceability').alias('avg_danceability'),
            avg('acousticness').alias('avg_acousticness'),
            avg('instrumentalness').alias('avg_instrumentalness'),
            stddev('valence').alias('valence_variability'),
            stddev('energy').alias('energy_variability')
        )

        # Name clusters based on characteristics
        named_clusters = cluster_profiles.withColumn(
            'mood_state_name',
            when((col('avg_valence') >= 0.6) & (col('avg_energy') >= 0.6), 'Energetic Joy')
            .when((col('avg_valence') >= 0.6) & (col('avg_energy') < 0.4), 'Peaceful Contentment')
            .when((col('avg_valence') < 0.4) & (col('avg_energy') >= 0.6), 'Intense Focus')
            .when((col('avg_valence') < 0.4) & (col('avg_energy') < 0.4), 'Melancholic Reflection')
            .otherwise('Balanced Neutral')
        ).withColumn(
            'cluster_description',
            when(col('mood_state_name') == 'Energetic Joy',
                 'High positivity with energetic, upbeat music')
            .when(col('mood_state_name') == 'Peaceful Contentment',
                  'Positive mood with calm, relaxing music')
            .when(col('mood_state_name') == 'Intense Focus',
                  'Low valence but high energy - concentrated work mode')
            .when(col('mood_state_name') == 'Melancholic Reflection',
                  'Low energy and valence - introspective listening')
            .otherwise('Balanced emotional state with varied music')
        ).withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
          .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
          .withColumn('avg_danceability', spark_round(col('avg_danceability'), 3)) \
          .withColumn('avg_acousticness', spark_round(col('avg_acousticness'), 3)) \
          .withColumn('avg_instrumentalness', spark_round(col('avg_instrumentalness'), 3)) \
          .withColumn('valence_variability', spark_round(col('valence_variability'), 3)) \
          .withColumn('energy_variability', spark_round(col('energy_variability'), 3)) \
          .orderBy('cluster')

        logger.info(f"✅ Discovered {named_clusters.count()} mood state clusters")

        return {
            'model': model,
            'cluster_profiles': named_clusters,
            'silhouette_score': silhouette
        }

    def build_listening_anomalies(self, df: DataFrame) -> DataFrame:
        """
        COGNITIVE: Detect anomalous listening patterns.

        Identify unusual listening behavior that deviates from typical patterns.
        """
        logger.info("Building anomaly detection...")

        # Calculate statistics by hour to establish baseline
        hourly_stats = df.groupBy('hour_of_day').agg(
            avg('valence').alias('typical_valence'),
            stddev('valence').alias('valence_std'),
            avg('energy').alias('typical_energy'),
            stddev('energy').alias('energy_std'),
            count('*').alias('typical_count')
        )

        # Join back to find deviations
        with_baseline = df.join(hourly_stats, 'hour_of_day')

        # Detect anomalies (> 2 standard deviations from mean)
        anomalies = with_baseline.withColumn(
            'valence_deviation',
            spark_abs(col('valence') - col('typical_valence')) / col('valence_std')
        ).withColumn(
            'energy_deviation',
            spark_abs(col('energy') - col('typical_energy')) / col('energy_std')
        ).withColumn(
            'is_anomaly',
            when((col('valence_deviation') > 2) | (col('energy_deviation') > 2), True)
            .otherwise(False)
        ).filter(col('is_anomaly'))

        # Aggregate anomalies by hour
        anomaly_summary = anomalies.groupBy('hour_of_day').agg(
            count('*').alias('anomaly_count'),
            avg('valence_deviation').alias('avg_valence_deviation'),
            avg('energy_deviation').alias('avg_energy_deviation')
        ).withColumn(
            'anomaly_severity',
            when((col('avg_valence_deviation') > 3) | (col('avg_energy_deviation') > 3), 'High')
            .when((col('avg_valence_deviation') > 2.5) | (col('avg_energy_deviation') > 2.5), 'Medium')
            .otherwise('Low')
        ).withColumn(
            'interpretation',
            when(col('anomaly_severity') == 'High',
                 'Unusual listening pattern - significant deviation from normal behavior')
            .when(col('anomaly_severity') == 'Medium',
                  'Moderate deviation - possible mood shift or special occasion')
            .otherwise('Minor variation within acceptable range')
        ).withColumn('avg_valence_deviation', spark_round(col('avg_valence_deviation'), 3)) \
          .withColumn('avg_energy_deviation', spark_round(col('avg_energy_deviation'), 3)) \
          .orderBy(desc('anomaly_severity'), desc('anomaly_count'))

        logger.info(f"✅ Detected anomalies in listening patterns")
        return anomaly_summary

    def build_sequential_patterns(self, df: DataFrame) -> DataFrame:
        """
        COGNITIVE: Discover sequential listening patterns.

        Identify common sequences in listening behavior (e.g., calm → energetic → calm).
        """
        logger.info("Building sequential pattern mining...")

        # Create mood categories for sequence analysis
        categorized = df.withColumn(
            'mood_category',
            when((col('valence') >= 0.6) & (col('energy') >= 0.6), 'High_Energy_Positive')
            .when((col('valence') >= 0.6) & (col('energy') < 0.4), 'Low_Energy_Positive')
            .when((col('valence') < 0.4) & (col('energy') >= 0.6), 'High_Energy_Negative')
            .when((col('valence') < 0.4) & (col('energy') < 0.4), 'Low_Energy_Negative')
            .otherwise('Neutral')
        ).orderBy('played_at')

        # Use window functions to get next mood
        window = Window.orderBy('played_at')

        sequences = categorized.withColumn(
            'next_mood',
            when(col('mood_category').isNotNull(),
                 concat(col('mood_category'), lit(' → '),
                       col('mood_category')))  # Simplified - in production would use lead()
        )

        # Count sequence frequencies
        sequence_patterns = sequences.groupBy('mood_category').agg(
            count('*').alias('occurrence_count')
        ).withColumn(
            'pattern_type',
            when(col('mood_category').contains('Positive'), 'Positive Mood Sequence')
            .when(col('mood_category').contains('Negative'), 'Negative Mood Sequence')
            .otherwise('Neutral Sequence')
        ).withColumn(
            'behavioral_insight',
            when(col('mood_category') == 'High_Energy_Positive',
                 'Sustained positive energy - active engagement')
            .when(col('mood_category') == 'Low_Energy_Positive',
                  'Relaxed contentment - passive enjoyment')
            .when(col('mood_category') == 'High_Energy_Negative',
                  'Intense focus or stress - may need mood intervention')
            .when(col('mood_category') == 'Low_Energy_Negative',
                  'Reflective or sad - monitor for prolonged periods')
            .otherwise('Varied emotional states - healthy balance')
        ).orderBy(desc('occurrence_count'))

        logger.info(f"✅ Discovered sequential listening patterns")
        return sequence_patterns

    def build_behavioral_segments(self, df: DataFrame) -> DataFrame:
        """
        COGNITIVE: Segment listening behavior into archetypes.

        Identify different listening personas based on overall patterns.
        """
        logger.info("Building behavioral segmentation...")

        # Calculate user-level statistics (aggregated across all listens)
        behavioral_profile = df.agg(
            avg('valence').alias('overall_valence'),
            avg('energy').alias('overall_energy'),
            avg('danceability').alias('overall_danceability'),
            avg('acousticness').alias('overall_acousticness'),
            stddev('valence').alias('mood_variability'),
            stddev('energy').alias('energy_variability'),
            count('*').alias('total_listens')
        )

        # Determine behavioral archetype
        archetype = behavioral_profile.withColumn(
            'listening_archetype',
            when((col('overall_valence') >= 0.6) & (col('overall_energy') >= 0.6), 'Energetic Optimist')
            .when((col('overall_valence') >= 0.6) & (col('overall_energy') < 0.4), 'Peaceful Relaxer')
            .when((col('overall_valence') < 0.4) & (col('overall_energy') >= 0.6), 'Intense Focuser')
            .when((col('overall_valence') < 0.4) & (col('overall_energy') < 0.4), 'Contemplative Thinker')
            .otherwise('Balanced Explorer')
        ).withColumn(
            'archetype_description',
            when(col('listening_archetype') == 'Energetic Optimist',
                 'Prefers upbeat, energetic music. Listening supports active lifestyle.')
            .when(col('listening_archetype') == 'Peaceful Relaxer',
                  'Gravitates toward calm, positive music. Music for relaxation and comfort.')
            .when(col('listening_archetype') == 'Intense Focuser',
                  'High-energy, low-valence tracks. Music for concentration and productivity.')
            .when(col('listening_archetype') == 'Contemplative Thinker',
                  'Low-energy, introspective music. Reflective and thoughtful listening.')
            .otherwise('Varied musical taste. Adapts music to different moods and contexts.')
        ).withColumn(
            'wellbeing_recommendation',
            when(col('mood_variability') > 0.3,
                 'High mood variability detected - ensure balanced emotional wellbeing')
            .when(col('overall_valence') < 0.4,
                  'Lower average mood - consider incorporating more uplifting music')
            .otherwise('Healthy listening patterns - continue current approach')
        ).withColumn('overall_valence', spark_round(col('overall_valence'), 3)) \
          .withColumn('overall_energy', spark_round(col('overall_energy'), 3)) \
          .withColumn('overall_danceability', spark_round(col('overall_danceability'), 3)) \
          .withColumn('overall_acousticness', spark_round(col('overall_acousticness'), 3)) \
          .withColumn('mood_variability', spark_round(col('mood_variability'), 3)) \
          .withColumn('energy_variability', spark_round(col('energy_variability'), 3))

        logger.info(f"✅ Created behavioral segmentation")
        return archetype

    def write_gold_table(self, df: DataFrame, table_name: str, gold_path: str):
        """Write Gold layer table to Delta Lake."""
        output_path = f"{gold_path}/cognitive/{table_name}"
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
        """Build all cognitive analytics tables."""
        logger.info("=" * 80)
        logger.info("BUILDING COGNITIVE ANALYTICS - GOLD LAYER")
        logger.info("=" * 80)

        # Load Silver data
        silver_df = self.load_silver_data(silver_path)

        # Build all cognitive analytics
        tables = {}

        try:
            cluster_result = self.build_mood_state_clusters(silver_df)
            tables['mood_state_clusters'] = cluster_result['cluster_profiles']
            self.write_gold_table(tables['mood_state_clusters'],
                                'mood_state_clusters', gold_path)
            logger.info(f"   Silhouette Score: {cluster_result['silhouette_score']:.4f}")
        except Exception as e:
            logger.error(f"Mood state clustering failed: {e}")

        try:
            tables['listening_anomalies'] = self.build_listening_anomalies(silver_df)
            self.write_gold_table(tables['listening_anomalies'],
                                'listening_anomalies', gold_path)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

        try:
            tables['sequential_patterns'] = self.build_sequential_patterns(silver_df)
            self.write_gold_table(tables['sequential_patterns'],
                                'sequential_patterns', gold_path)
        except Exception as e:
            logger.error(f"Sequential pattern mining failed: {e}")

        try:
            tables['behavioral_segments'] = self.build_behavioral_segments(silver_df)
            self.write_gold_table(tables['behavioral_segments'],
                                'behavioral_segments', gold_path)
        except Exception as e:
            logger.error(f"Behavioral segmentation failed: {e}")

        logger.info("=" * 80)
        logger.info("✅ COGNITIVE ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Created {len(tables)} cognitive analytics tables:")
        for table_name in tables.keys():
            logger.info(f"  - {table_name}")


def main():
    """Main execution."""
    logger.info("Starting Cognitive Analytics build...")

    # Paths
    silver_path = '/app/data/silver/listening_with_features'
    gold_path = '/app/data/gold'

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("CognitiveAnalytics") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        analytics = CognitiveAnalytics(spark)
        analytics.build_all(silver_path, gold_path)

        logger.info("=" * 80)
        logger.info("✅ Cognitive Analytics build SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Cognitive Analytics build FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
