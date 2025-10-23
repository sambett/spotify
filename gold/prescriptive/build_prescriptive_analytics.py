"""
PRESCRIPTIVE ANALYTICS - Gold Layer

Creates actionable recommendations based on predictive insights:
1. Track recommendations for mood improvement
2. Optimal listening times for different moods
3. Personalized playlist suggestions
4. Mood-based intervention suggestions

These tables answer: "What should we do?" and "What actions to take?"
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, avg, min, max, desc, row_number,
    when, round as spark_round, abs as spark_abs, explode, array, lit, struct
)

from utils.logger import setup_logger

logger = setup_logger('prescriptive_analytics')


class PrescriptiveAnalytics:
    """Build prescriptive analytics tables from Silver and Predictive layers."""

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

    def load_mood_predictions(self, predictions_path: str) -> DataFrame:
        """Load mood predictions from Predictive layer."""
        logger.info(f"Loading mood predictions from: {predictions_path}")
        try:
            df = self.spark.read.format('delta').load(predictions_path)
            logger.info(f"✅ Loaded mood predictions")
            return df
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            raise

    def build_mood_improvement_recommendations(self, df: DataFrame) -> DataFrame:
        """
        PRESCRIPTIVE: Recommend tracks to improve mood (increase valence).

        For low-mood periods, recommend high-valence tracks that were successful.
        """
        logger.info("Building mood improvement recommendations...")

        # Identify low-mood periods (valence < 0.4)
        low_mood_periods = df.filter(col('valence') < 0.4) \
            .select('hour_of_day', 'part_of_day') \
            .distinct()

        # Find high-valence tracks (valence >= 0.7) that boost mood
        high_valence_tracks = df.filter(col('valence') >= 0.7) \
            .groupBy('track_name', 'artist_name') \
            .agg(
                avg('valence').alias('avg_valence'),
                avg('energy').alias('avg_energy'),
                avg('danceability').alias('avg_danceability'),
                count('*').alias('play_count')
            ) \
            .filter(col('play_count') >= 2) \
            .orderBy(desc('avg_valence')) \
            .limit(20)

        # Create recommendations
        recommendations = high_valence_tracks.withColumn(
            'recommendation_reason',
            when(col('avg_energy') >= 0.7, 'High energy track to boost mood')
            .when(col('avg_danceability') >= 0.7, 'Danceable track to improve mood')
            .otherwise('Positive track to enhance wellbeing')
        ).withColumn(
            'target_scenario',
            lit('When feeling down or low energy')
        ).withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
          .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
          .withColumn('avg_danceability', spark_round(col('avg_danceability'), 3)) \
          .select(
              'track_name',
              'artist_name',
              'avg_valence',
              'avg_energy',
              'avg_danceability',
              'play_count',
              'recommendation_reason',
              'target_scenario'
          )

        logger.info(f"✅ Created mood improvement recommendations")
        return recommendations

    def build_optimal_listening_times(self, df: DataFrame, predictions_df: DataFrame) -> DataFrame:
        """
        PRESCRIPTIVE: Suggest optimal times to listen for desired mood states.

        Based on predictions, recommend when to listen for specific outcomes.
        """
        logger.info("Building optimal listening time recommendations...")

        # Calculate average predicted mood by hour
        optimal_times = predictions_df.groupBy('hour_of_day').agg(
            avg('predicted_valence').alias('predicted_valence'),
            avg('energy').alias('avg_energy')
        )

        # Add recommendations based on predicted mood
        recommendations = optimal_times.withColumn(
            'mood_state',
            when((col('predicted_valence') >= 0.6) & (col('avg_energy') >= 0.6), 'Happy & Energetic')
            .when((col('predicted_valence') >= 0.6) & (col('avg_energy') < 0.4), 'Happy & Calm')
            .when((col('predicted_valence') < 0.4) & (col('avg_energy') >= 0.6), 'Intense & Focused')
            .when((col('predicted_valence') < 0.4) & (col('avg_energy') < 0.4), 'Reflective & Calm')
            .otherwise('Neutral')
        ).withColumn(
            'recommended_activity',
            when(col('mood_state') == 'Happy & Energetic', 'Exercise, social activities, upbeat playlists')
            .when(col('mood_state') == 'Happy & Calm', 'Relaxation, acoustic music, mindfulness')
            .when(col('mood_state') == 'Intense & Focused', 'Deep work, instrumental music, concentration')
            .when(col('mood_state') == 'Reflective & Calm', 'Journaling, ambient music, meditation')
            .otherwise('General listening, explore new music')
        ).withColumn(
            'wellbeing_tip',
            when(col('mood_state') == 'Happy & Energetic', 'Great time for physical activity and social connection')
            .when(col('mood_state') == 'Happy & Calm', 'Perfect for self-care and peaceful activities')
            .when(col('mood_state') == 'Intense & Focused', 'Ideal for productivity and focused work')
            .when(col('mood_state') == 'Reflective & Calm', 'Good for introspection and quiet contemplation')
            .otherwise('Balance your mood with varied activities')
        ).withColumn('predicted_valence', spark_round(col('predicted_valence'), 3)) \
          .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
          .select(
              'hour_of_day',
              'mood_state',
              'predicted_valence',
              'avg_energy',
              'recommended_activity',
              'wellbeing_tip'
          ).orderBy('hour_of_day')

        logger.info(f"✅ Created optimal listening time recommendations")
        return recommendations

    def build_personalized_playlist_suggestions(self, df: DataFrame) -> DataFrame:
        """
        PRESCRIPTIVE: Generate playlist suggestions for different use cases.

        Create playlists based on desired outcomes (focus, relaxation, energy, etc.)
        """
        logger.info("Building personalized playlist suggestions...")

        # Define playlist types and their criteria
        playlists = []

        # Playlist 1: Focus & Concentration (low valence, moderate energy, high instrumentalness)
        focus_tracks = df.filter(
            (col('instrumentalness') >= 0.5) &
            (col('energy').between(0.4, 0.7)) &
            (col('tempo').between(90, 130))
        ).groupBy('track_name', 'artist_name').agg(
            avg('instrumentalness').alias('avg_instrumentalness'),
            avg('energy').alias('avg_energy'),
            count('*').alias('play_count')
        ).orderBy(desc('avg_instrumentalness')).limit(10) \
          .withColumn('playlist_name', lit('Focus & Deep Work')) \
          .withColumn('playlist_purpose', lit('Enhance concentration and productivity'))

        # Playlist 2: Energy Boost (high energy, high valence, high tempo)
        energy_tracks = df.filter(
            (col('energy') >= 0.7) &
            (col('valence') >= 0.6) &
            (col('tempo') >= 120)
        ).groupBy('track_name', 'artist_name').agg(
            avg('energy').alias('avg_energy'),
            avg('valence').alias('avg_valence'),
            count('*').alias('play_count')
        ).orderBy(desc('avg_energy')).limit(10) \
          .withColumn('playlist_name', lit('Energy Boost')) \
          .withColumn('playlist_purpose', lit('Increase energy and motivation'))

        # Playlist 3: Relaxation (high acousticness, low energy, moderate valence)
        relax_tracks = df.filter(
            (col('acousticness') >= 0.6) &
            (col('energy') < 0.5) &
            (col('valence').between(0.4, 0.7))
        ).groupBy('track_name', 'artist_name').agg(
            avg('acousticness').alias('avg_acousticness'),
            avg('energy').alias('avg_energy'),
            count('*').alias('play_count')
        ).orderBy(desc('avg_acousticness')).limit(10) \
          .withColumn('playlist_name', lit('Relaxation & Calm')) \
          .withColumn('playlist_purpose', lit('Reduce stress and promote relaxation'))

        # Playlist 4: Mood Lift (high valence, moderate-high energy, high danceability)
        mood_lift_tracks = df.filter(
            (col('valence') >= 0.7) &
            (col('danceability') >= 0.6) &
            (col('energy') >= 0.5)
        ).groupBy('track_name', 'artist_name').agg(
            avg('valence').alias('avg_valence'),
            avg('danceability').alias('avg_danceability'),
            count('*').alias('play_count')
        ).orderBy(desc('avg_valence')).limit(10) \
          .withColumn('playlist_name', lit('Mood Lift')) \
          .withColumn('playlist_purpose', lit('Improve mood and emotional wellbeing'))

        # Union all playlists
        all_playlists = focus_tracks.select('playlist_name', 'playlist_purpose', 'track_name', 'artist_name', 'play_count') \
            .union(energy_tracks.select('playlist_name', 'playlist_purpose', 'track_name', 'artist_name', 'play_count')) \
            .union(relax_tracks.select('playlist_name', 'playlist_purpose', 'track_name', 'artist_name', 'play_count')) \
            .union(mood_lift_tracks.select('playlist_name', 'playlist_purpose', 'track_name', 'artist_name', 'play_count'))

        logger.info(f"✅ Created personalized playlist suggestions")
        return all_playlists

    def build_mood_intervention_triggers(self, df: DataFrame) -> DataFrame:
        """
        PRESCRIPTIVE: Identify when intervention is needed for wellbeing.

        Detect patterns that suggest mood support interventions.
        """
        logger.info("Building mood intervention triggers...")

        # Analyze consecutive low-valence periods
        window = Window.orderBy('played_at')

        mood_patterns = df.withColumn('prev_valence', col('valence')) \
            .withColumn('is_low_mood', when(col('valence') < 0.3, 1).otherwise(0))

        # Aggregate by hour to find concerning patterns
        intervention_triggers = mood_patterns.groupBy('hour_of_day').agg(
            avg('valence').alias('avg_valence'),
            avg('energy').alias('avg_energy'),
            count(when(col('is_low_mood') == 1, 1)).alias('low_mood_count'),
            count('*').alias('total_plays')
        ).withColumn(
            'intervention_needed',
            when(col('avg_valence') < 0.3, 'High Priority')
            .when(col('avg_valence') < 0.4, 'Medium Priority')
            .otherwise('Low Priority')
        ).withColumn(
            'suggested_intervention',
            when(col('avg_valence') < 0.3, 'Consider mood-boosting activities, reach out to support network')
            .when(col('avg_valence') < 0.4, 'Listen to uplifting music, engage in light physical activity')
            .otherwise('Continue normal listening patterns')
        ).withColumn(
            'mood_support_playlist',
            when(col('avg_valence') < 0.4, 'Energy Boost, Mood Lift')
            .otherwise('Personalized Mix')
        ).withColumn('avg_valence', spark_round(col('avg_valence'), 3)) \
          .withColumn('avg_energy', spark_round(col('avg_energy'), 3)) \
          .filter(col('intervention_needed') != 'Low Priority') \
          .select(
              'hour_of_day',
              'avg_valence',
              'avg_energy',
              'low_mood_count',
              'total_plays',
              'intervention_needed',
              'suggested_intervention',
              'mood_support_playlist'
          ).orderBy('avg_valence')

        logger.info(f"✅ Created mood intervention triggers")
        return intervention_triggers

    def write_gold_table(self, df: DataFrame, table_name: str, gold_path: str):
        """Write Gold layer table to Delta Lake."""
        output_path = f"{gold_path}/prescriptive/{table_name}"
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

    def build_all(self, silver_path: str, predictions_path: str, gold_path: str):
        """Build all prescriptive analytics tables."""
        logger.info("=" * 80)
        logger.info("BUILDING PRESCRIPTIVE ANALYTICS - GOLD LAYER")
        logger.info("=" * 80)

        # Load data
        silver_df = self.load_silver_data(silver_path)
        predictions_df = self.load_mood_predictions(predictions_path)

        # Build all prescriptive tables
        tables = {}

        try:
            tables['mood_improvement_recommendations'] = self.build_mood_improvement_recommendations(silver_df)
            self.write_gold_table(tables['mood_improvement_recommendations'],
                                'mood_improvement_recommendations', gold_path)
        except Exception as e:
            logger.error(f"Mood improvement recommendations failed: {e}")

        try:
            tables['optimal_listening_times'] = self.build_optimal_listening_times(silver_df, predictions_df)
            self.write_gold_table(tables['optimal_listening_times'],
                                'optimal_listening_times', gold_path)
        except Exception as e:
            logger.error(f"Optimal listening times failed: {e}")

        try:
            tables['personalized_playlist_suggestions'] = self.build_personalized_playlist_suggestions(silver_df)
            self.write_gold_table(tables['personalized_playlist_suggestions'],
                                'personalized_playlist_suggestions', gold_path)
        except Exception as e:
            logger.error(f"Playlist suggestions failed: {e}")

        try:
            tables['mood_intervention_triggers'] = self.build_mood_intervention_triggers(silver_df)
            self.write_gold_table(tables['mood_intervention_triggers'],
                                'mood_intervention_triggers', gold_path)
        except Exception as e:
            logger.error(f"Intervention triggers failed: {e}")

        logger.info("=" * 80)
        logger.info("✅ PRESCRIPTIVE ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Created {len(tables)} prescriptive tables:")
        for table_name in tables.keys():
            logger.info(f"  - {table_name}")


def main():
    """Main execution."""
    logger.info("Starting Prescriptive Analytics build...")

    # Paths
    silver_path = '/app/data/silver/listening_with_features'
    predictions_path = '/app/data/gold/predictive/mood_predictions'
    gold_path = '/app/data/gold'

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("PrescriptiveAnalytics") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        analytics = PrescriptiveAnalytics(spark)
        analytics.build_all(silver_path, predictions_path, gold_path)

        logger.info("=" * 80)
        logger.info("✅ Prescriptive Analytics build SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Prescriptive Analytics build FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
