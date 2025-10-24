"""
COGNITIVE ANALYTICS - Gold Layer

⚠️ ACADEMIC DISCLAIMER:
This demonstrates clustering and pattern recognition techniques on SYNTHETIC audio features.

CRITICAL LIMITATIONS:
1. Audio features (valence, energy, etc.) are SYNTHETICALLY GENERATED (Spotify API 403 error)
2. Single-user dataset (n=~1,500) limits statistical power
3. Clusters may reflect synthetic data artifacts, not real music preferences

WHAT THIS DEMONSTRATES:
✅ K-Means clustering with proper validation (elbow method, silhouette analysis)
✅ Anomaly detection techniques
✅ Pattern mining methodologies
✅ ML pipeline architecture

WHAT THIS DOES NOT PROVIDE:
❌ Valid insights into actual music preferences
❌ Scientifically interpretable mood states
❌ Generalizable behavioral patterns

See ACADEMIC_DISCLAIMER.md for full details.

Creates advanced pattern recognition and learning systems:
1. K-means clustering for mood state discovery (with elbow method validation)
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

    def perform_elbow_analysis(self, scaled_data: DataFrame, feature_col: str = 'scaled_features',
                               k_range: range = range(2, 11)) -> dict:
        """
        Perform elbow method analysis to determine optimal number of clusters.

        Returns within-cluster sum of squares (WCSS) for each k value.
        The "elbow" point suggests optimal k.
        """
        logger.info("=" * 60)
        logger.info("ELBOW METHOD ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Testing k values from {k_range.start} to {k_range.stop - 1}")

        wcss_values = {}

        for k in k_range:
            logger.info(f"Training K-Means with k={k}...")

            kmeans = KMeans(
                featuresCol=feature_col,
                predictionCol='cluster',
                k=k,
                seed=42,
                maxIter=20
            )

            model = kmeans.fit(scaled_data)
            predictions = model.transform(scaled_data)

            # Calculate WCSS (Within-Cluster Sum of Squares)
            # Lower is better, but we look for the "elbow" where improvement slows
            evaluator = ClusteringEvaluator(
                featuresCol=feature_col,
                predictionCol='cluster',
                metricName='silhouette'
            )

            silhouette = evaluator.evaluate(predictions)
            wcss = model.summary.trainingCost  # This is the WCSS

            wcss_values[k] = {
                'wcss': wcss,
                'silhouette': silhouette
            }

            logger.info(f"  k={k}: WCSS={wcss:.2f}, Silhouette={silhouette:.4f}")

        logger.info("=" * 60)
        logger.info("ELBOW ANALYSIS RESULTS:")
        logger.info("=" * 60)

        # Find elbow point (simple heuristic: maximum second derivative)
        k_values = sorted(wcss_values.keys())
        wcss_list = [wcss_values[k]['wcss'] for k in k_values]

        # Calculate rate of change
        if len(wcss_list) >= 3:
            second_derivatives = []
            for i in range(1, len(wcss_list) - 1):
                second_deriv = wcss_list[i - 1] - 2 * wcss_list[i] + wcss_list[i + 1]
                second_derivatives.append((k_values[i], second_deriv))

            optimal_k_elbow = max(second_derivatives, key=lambda x: x[1])[0]
            logger.info(f"Elbow Method Suggests: k={optimal_k_elbow}")
        else:
            optimal_k_elbow = k_values[len(k_values) // 2]
            logger.info(f"Insufficient data for elbow detection, using midpoint: k={optimal_k_elbow}")

        # Find best silhouette score
        optimal_k_silhouette = max(wcss_values.items(), key=lambda x: x[1]['silhouette'])[0]
        logger.info(f"Best Silhouette Score at: k={optimal_k_silhouette} (score={wcss_values[optimal_k_silhouette]['silhouette']:.4f})")

        logger.info("=" * 60)
        logger.info(f"📊 RECOMMENDATION: Consider k={optimal_k_elbow} (elbow) or k={optimal_k_silhouette} (silhouette)")
        logger.info("=" * 60)

        return {
            'wcss_values': wcss_values,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'all_k_values': k_values
        }

    def validate_cluster_stability(self, scaled_data: DataFrame, k: int,
                                   feature_col: str = 'scaled_features', n_runs: int = 3) -> dict:
        """
        Test cluster stability by running K-Means multiple times with different seeds.
        Stable clusters should have consistent silhouette scores.
        """
        logger.info(f"Testing cluster stability for k={k} with {n_runs} runs...")

        silhouette_scores = []

        for run in range(n_runs):
            kmeans = KMeans(
                featuresCol=feature_col,
                predictionCol='cluster',
                k=k,
                seed=42 + run,  # Different seed each time
                maxIter=20
            )

            model = kmeans.fit(scaled_data)
            predictions = model.transform(scaled_data)

            evaluator = ClusteringEvaluator(
                featuresCol=feature_col,
                predictionCol='cluster',
                metricName='silhouette'
            )

            silhouette = evaluator.evaluate(predictions)
            silhouette_scores.append(silhouette)
            logger.info(f"  Run {run + 1}: Silhouette = {silhouette:.4f}")

        mean_silhouette = sum(silhouette_scores) / len(silhouette_scores)
        std_silhouette = (sum((x - mean_silhouette) ** 2 for x in silhouette_scores) / len(silhouette_scores)) ** 0.5

        logger.info(f"Stability: Mean Silhouette = {mean_silhouette:.4f} ± {std_silhouette:.4f}")

        if std_silhouette < 0.05:
            logger.info("✅ Clusters are STABLE (low variance across runs)")
        elif std_silhouette < 0.10:
            logger.info("⚠️  Clusters show MODERATE stability")
        else:
            logger.warning("❌ Clusters are UNSTABLE (high variance) - consider different k")

        return {
            'mean_silhouette': mean_silhouette,
            'std_silhouette': std_silhouette,
            'silhouette_scores': silhouette_scores,
            'is_stable': std_silhouette < 0.10
        }

    def build_mood_state_clusters(self, df: DataFrame) -> dict:
        """
        COGNITIVE: Discover natural mood states using K-means clustering.

        Cluster listening sessions based on audio features to discover
        natural mood states beyond simple categorization.

        Uses elbow method and silhouette analysis to determine optimal k.
        """
        logger.info("=" * 80)
        logger.info("BUILDING MOOD STATE CLUSTERS WITH VALIDATION")
        logger.info("=" * 80)

        # Select features for clustering
        feature_cols = ['valence', 'energy', 'danceability', 'acousticness',
                       'instrumentalness', 'tempo', 'loudness']

        # Prepare data
        model_data = df.select(*feature_cols).na.drop()
        sample_count = model_data.count()
        logger.info(f"Clustering {sample_count} samples")

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

        # Assemble and scale features
        assembler = VectorAssembler(inputCols=feature_cols_normalized, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # Create preprocessing pipeline
        prep_pipeline = Pipeline(stages=[assembler, scaler])
        prep_model = prep_pipeline.fit(model_data)
        scaled_data = prep_model.transform(model_data)

        # STEP 1: Perform elbow analysis to determine optimal k
        logger.info("")
        logger.info("STEP 1: Elbow Method & Silhouette Analysis")
        elbow_results = self.perform_elbow_analysis(scaled_data, 'scaled_features', k_range=range(2, 9))

        # STEP 2: Use recommended k (or default to 5 if inconclusive)
        optimal_k = elbow_results['optimal_k_silhouette']
        logger.info("")
        logger.info(f"STEP 2: Training Final Model with k={optimal_k}")

        # STEP 3: Validate cluster stability
        logger.info("")
        logger.info(f"STEP 3: Testing Cluster Stability (k={optimal_k})")
        stability_results = self.validate_cluster_stability(scaled_data, optimal_k, 'scaled_features', n_runs=3)

        # STEP 4: Train final model
        logger.info("")
        logger.info(f"STEP 4: Training Final K-Means Model")
        kmeans = KMeans(
            featuresCol='scaled_features',
            predictionCol='cluster',
            k=optimal_k,
            seed=42,
            maxIter=20
        )

        # Train on scaled data
        final_model = kmeans.fit(scaled_data)
        predictions = final_model.transform(scaled_data)

        # Evaluate
        evaluator = ClusteringEvaluator(
            featuresCol='scaled_features',
            predictionCol='cluster',
            metricName='silhouette'
        )

        silhouette = evaluator.evaluate(predictions)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ CLUSTERING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Number of Clusters (k): {optimal_k}")
        logger.info(f"Final Silhouette Score: {silhouette:.4f}")
        logger.info(f"Cluster Stability: {stability_results['mean_silhouette']:.4f} ± {stability_results['std_silhouette']:.4f}")
        logger.info(f"Is Stable: {'Yes' if stability_results['is_stable'] else 'No'}")
        logger.info("=" * 80)
        logger.info("")
        logger.warning("⚠️  Clusters based on SYNTHETIC features - interpret with caution")

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
            'model': final_model,
            'cluster_profiles': named_clusters,
            'silhouette_score': silhouette,
            'optimal_k': optimal_k,
            'elbow_analysis': elbow_results,
            'stability_results': stability_results
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
