"""
PREDICTIVE ANALYTICS - Gold Layer

CORRECTED APPROACH:
- Train on Kaggle dataset (114K tracks with real audio features)
- Test on user's listening history (1,504 events)
- This provides proper generalization testing

Creates machine learning models to forecast future patterns:
1. Mood prediction based on time features
2. Energy level forecasting
3. Mood category classification

These tables answer: "What will happen?" and "What are future trends?"

Uses Spark MLlib for scalable ML.
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, rand, abs as spark_abs, hour, dayofweek
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

from utils.logger import setup_logger

logger = setup_logger('predictive_analytics')


class PredictiveAnalytics:
    """Build predictive ML models trained on Kaggle, tested on real listening data."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_kaggle_training_data(self, bronze_path: str) -> DataFrame:
        """Load Kaggle dataset as TRAINING data (114K tracks with real features)."""
        logger.info(f"Loading Kaggle training data from: {bronze_path}")
        try:
            df = self.spark.read.format('delta').load(f"{bronze_path}/kaggle_tracks_bronze")
            count = df.count()
            logger.info(f"✅ Loaded {count} Kaggle tracks for TRAINING")

            # Add synthetic time features for training (since Kaggle doesn't have timestamps)
            # Use random but realistic distributions
            df = df.withColumn('hour_of_day', (rand(seed=42) * 24).cast('int'))
            df = df.withColumn('day_of_week', (rand(seed=43) * 7).cast('int'))
            df = df.withColumn('is_weekend',
                              when(col('day_of_week').isin([0, 6]), 1).otherwise(0))

            logger.info("Added synthetic time features to Kaggle training data")
            return df
        except Exception as e:
            logger.error(f"Failed to load Kaggle data: {e}")
            raise

    def load_listening_test_data(self, silver_path: str) -> DataFrame:
        """Load user's listening history as TEST data (1,504 events with real timestamps)."""
        logger.info(f"Loading listening test data from: {silver_path}")
        try:
            df = self.spark.read.format('delta').load(silver_path)
            count = df.count()
            logger.info(f"✅ Loaded {count} listening events for TESTING")

            # Ensure is_weekend is numeric
            if 'is_weekend' in df.columns:
                df = df.withColumn('is_weekend_int',
                                  when(col('is_weekend'), 1).otherwise(0))

            return df
        except Exception as e:
            logger.error(f"Failed to load listening test data: {e}")
            raise

    def build_mood_prediction_model(self, train_df: DataFrame, test_df: DataFrame) -> dict:
        """
        PREDICTIVE: Predict valence (mood) based on time and audio features.

        Train: Kaggle (114K tracks)
        Test: User's listening history (1,504 events)

        Model: Random Forest Regression
        Features: hour_of_day, day_of_week, is_weekend, energy, tempo, danceability
        Target: valence (happiness level)
        """
        logger.info("=" * 80)
        logger.info("Building mood prediction model...")
        logger.info("Train: Kaggle dataset | Test: User's listening history")
        logger.info("=" * 80)

        # Prepare features (common to both datasets)
        feature_cols = ['hour_of_day', 'day_of_week', 'energy', 'tempo', 'danceability']

        # Prepare training data
        train_select = feature_cols + ['valence', 'is_weekend']
        train_data = train_df.select(*train_select) \
            .filter(col('valence').isNotNull())

        for feat in feature_cols:
            train_data = train_data.filter(col(feat).isNotNull())

        train_data = train_data.withColumn('is_weekend_int', col('is_weekend'))
        feature_cols_final = feature_cols + ['is_weekend_int']

        # Prepare test data
        test_select = feature_cols + ['valence']
        if 'is_weekend_int' in test_df.columns:
            test_select.append('is_weekend_int')
        elif 'is_weekend' in test_df.columns:
            test_data = test_df.select(*feature_cols, 'valence', 'is_weekend')
            test_data = test_data.withColumn('is_weekend_int',
                                            when(col('is_weekend'), 1).otherwise(0))
        else:
            test_data = test_df.select(*feature_cols, 'valence')
            test_data = test_data.withColumn('is_weekend_int', 0)

        test_data = test_df.select(*test_select) if 'is_weekend_int' in test_df.columns else test_data
        test_data = test_data.filter(col('valence').isNotNull())

        for feat in feature_cols:
            test_data = test_data.filter(col(feat).isNotNull())

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols_final, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # Random Forest model
        rf = RandomForestRegressor(
            featuresCol='scaled_features',
            labelCol='valence',
            numTrees=30,
            maxDepth=8,
            minInstancesPerNode=10,
            seed=42
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, rf])

        train_count = train_data.count()
        test_count = test_data.count()
        logger.info(f"Training on {train_count} Kaggle tracks")
        logger.info(f"Testing on {test_count} user listening events")

        # Train on Kaggle
        model = pipeline.fit(train_data)

        # Test on user's listening history
        predictions = model.transform(test_data)

        # Evaluate
        evaluator = RegressionEvaluator(
            labelCol='valence',
            predictionCol='prediction',
            metricName='rmse'
        )

        rmse = evaluator.evaluate(predictions)
        mae_evaluator = RegressionEvaluator(labelCol='valence', predictionCol='prediction', metricName='mae')
        mae = mae_evaluator.evaluate(predictions)
        r2_evaluator = RegressionEvaluator(labelCol='valence', predictionCol='prediction', metricName='r2')
        r2 = r2_evaluator.evaluate(predictions)

        logger.info("=" * 80)
        logger.info(f"✅ Mood Prediction Model (Kaggle→User)")
        logger.info(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        logger.info("=" * 80)

        # Save predictions on user's listening data
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'valence', 'prediction', 'energy', 'tempo'
        ).withColumnRenamed('prediction', 'predicted_valence')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'train_size': train_count,
                'test_size': test_count
            },
            'model_type': 'RandomForestRegressor',
            'training_approach': 'Kaggle_to_User'
        }

    def build_energy_forecast_model(self, train_df: DataFrame, test_df: DataFrame) -> dict:
        """
        PREDICTIVE: Forecast energy levels.

        Train: Kaggle (114K tracks)
        Test: User's listening history (1,504 events)

        Model: Linear Regression
        Features: hour_of_day, day_of_week, tempo, danceability, loudness
        Target: energy
        """
        logger.info("=" * 80)
        logger.info("Building energy forecast model...")
        logger.info("Train: Kaggle dataset | Test: User's listening history")
        logger.info("=" * 80)

        # NOTE: Kaggle dataset doesn't have 'loudness', so we exclude it
        feature_cols = ['hour_of_day', 'day_of_week', 'tempo', 'danceability', 'acousticness']

        # Prepare training data
        train_data = train_df.select(*feature_cols, 'energy') \
            .filter(col('energy').isNotNull())

        for feat in feature_cols:
            train_data = train_data.filter(col(feat).isNotNull())

        # Prepare test data
        test_data = test_df.select(*feature_cols, 'energy') \
            .filter(col('energy').isNotNull())

        for feat in feature_cols:
            test_data = test_data.filter(col(feat).isNotNull())

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # Linear Regression model
        lr = LinearRegression(
            featuresCol='scaled_features',
            labelCol='energy',
            maxIter=100,
            regParam=0.01
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, lr])

        train_count = train_data.count()
        test_count = test_data.count()
        logger.info(f"Training on {train_count} Kaggle tracks")
        logger.info(f"Testing on {test_count} user listening events")

        # Train on Kaggle
        model = pipeline.fit(train_data)

        # Test on user's listening history
        predictions = model.transform(test_data)

        # Evaluate
        evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='rmse')
        rmse = evaluator.evaluate(predictions)
        mae_evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='mae')
        mae = mae_evaluator.evaluate(predictions)
        r2_evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='r2')
        r2 = r2_evaluator.evaluate(predictions)

        logger.info("=" * 80)
        logger.info(f"✅ Energy Forecast Model (Kaggle→User)")
        logger.info(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        logger.info("=" * 80)

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'energy', 'prediction', 'tempo', 'danceability'
        ).withColumnRenamed('prediction', 'predicted_energy')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'train_size': train_count,
                'test_size': test_count
            },
            'model_type': 'LinearRegression',
            'training_approach': 'Kaggle_to_User'
        }

    def build_mood_category_classifier(self, train_df: DataFrame, test_df: DataFrame) -> dict:
        """
        PREDICTIVE: Classify listening sessions into mood categories.

        Train: Kaggle (114K tracks)
        Test: User's listening history (1,504 events)

        Model: Random Forest Classifier
        Features: hour_of_day, tempo, acousticness, danceability, loudness, speechiness
        Target: mood_category (Happy_Energetic, Happy_Calm, etc.)
        """
        logger.info("=" * 80)
        logger.info("Building mood category classifier...")
        logger.info("Train: Kaggle dataset | Test: User's listening history")
        logger.info("=" * 80)

        # Create mood categories based on valence and energy
        # NOTE: We EXCLUDE valence and energy from features to avoid data leakage
        def add_mood_labels(df: DataFrame) -> DataFrame:
            return df.withColumn(
                'mood_label',
                when((col('valence') >= 0.6) & (col('energy') >= 0.6), 0)  # Happy_Energetic
                .when((col('valence') >= 0.6) & (col('energy') < 0.4), 1)   # Happy_Calm
                .when((col('valence') < 0.4) & (col('energy') >= 0.6), 2)   # Sad_Energetic
                .when((col('valence') < 0.4) & (col('energy') < 0.4), 3)    # Sad_Calm
                .otherwise(4)  # Neutral
            )

        train_mood_data = add_mood_labels(train_df)
        test_mood_data = add_mood_labels(test_df)

        # Features: Use only external/temporal features, NOT valence/energy (to avoid leakage)
        # NOTE: Kaggle dataset doesn't have 'loudness' or 'speechiness', so we exclude them
        feature_cols = ['hour_of_day', 'day_of_week', 'tempo', 'danceability', 'acousticness', 'instrumentalness']

        # Filter nulls
        train_data = train_mood_data.select(*feature_cols, 'mood_label') \
            .filter(col('mood_label').isNotNull())

        for feat in feature_cols:
            train_data = train_data.filter(col(feat).isNotNull())

        test_data = test_mood_data.select(*feature_cols, 'mood_label') \
            .filter(col('mood_label').isNotNull())

        for feat in feature_cols:
            test_data = test_data.filter(col(feat).isNotNull())

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # Random Forest Classifier
        rf_classifier = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol='mood_label',
            numTrees=30,
            maxDepth=8,
            minInstancesPerNode=10,
            seed=42
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, rf_classifier])

        train_count = train_data.count()
        test_count = test_data.count()
        logger.info(f"Training on {train_count} Kaggle tracks")
        logger.info(f"Testing on {test_count} user listening events")

        # Train on Kaggle
        model = pipeline.fit(train_data)

        # Test on user's listening history
        predictions = model.transform(test_data)

        # Evaluate
        evaluator = MulticlassClassificationEvaluator(
            labelCol='mood_label',
            predictionCol='prediction',
            metricName='accuracy'
        )

        accuracy = evaluator.evaluate(predictions)

        f1_evaluator = MulticlassClassificationEvaluator(labelCol='mood_label', predictionCol='prediction', metricName='f1')
        f1 = f1_evaluator.evaluate(predictions)

        logger.info("=" * 80)
        logger.info(f"✅ Mood Category Classifier (Kaggle→User)")
        logger.info(f"   Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
        logger.info("=" * 80)

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'mood_label', 'prediction'
        ).withColumnRenamed('prediction', 'predicted_mood_label')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {
                'accuracy': accuracy,
                'f1': f1,
                'train_size': train_count,
                'test_size': test_count
            },
            'model_type': 'RandomForestClassifier',
            'training_approach': 'Kaggle_to_User'
        }

    def write_gold_table(self, df: DataFrame, table_name: str, gold_path: str):
        """Write Gold layer table to Delta Lake."""
        output_path = f"{gold_path}/predictive/{table_name}"
        logger.info(f"Writing {table_name} to: {output_path}")

        try:
            df.write \
                .format('delta') \
                .mode('overwrite') \
                .option('overwriteSchema', 'true') \
                .save(output_path)

            logger.info(f"✅ Successfully wrote {table_name}")
        except Exception as e:
            logger.error(f"Failed to write {table_name}: {e}")
            raise

    def save_model_metrics(self, models: dict, gold_path: str):
        """Save model performance metrics."""
        logger.info("Saving model metrics...")

        metrics_data = []
        for model_name, model_info in models.items():
            metrics = model_info['metrics']
            metrics_row = {
                'model_name': model_name,
                'model_type': model_info['model_type'],
                'training_approach': model_info.get('training_approach', 'Unknown')
            }
            metrics_row.update(metrics)
            metrics_data.append(metrics_row)

        metrics_df = self.spark.createDataFrame(metrics_data)
        self.write_gold_table(metrics_df, 'model_performance_metrics', gold_path)

    def build_all(self, bronze_path: str, silver_path: str, gold_path: str):
        """Build all predictive analytics models."""
        logger.info("=" * 80)
        logger.info("BUILDING PREDICTIVE ANALYTICS - GOLD LAYER")
        logger.info("CORRECTED APPROACH: Train on Kaggle, Test on User Data")
        logger.info("=" * 80)

        # Load Kaggle as training data
        kaggle_train = self.load_kaggle_training_data(bronze_path)

        # Load user's listening history as test data
        listening_test = self.load_listening_test_data(silver_path)

        # Build all predictive models
        models = {}

        try:
            models['mood_prediction'] = self.build_mood_prediction_model(kaggle_train, listening_test)
            self.write_gold_table(models['mood_prediction']['predictions'],
                                'mood_predictions', gold_path)
        except Exception as e:
            logger.error(f"Mood prediction model failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        try:
            models['energy_forecast'] = self.build_energy_forecast_model(kaggle_train, listening_test)
            self.write_gold_table(models['energy_forecast']['predictions'],
                                'energy_forecasts', gold_path)
        except Exception as e:
            logger.error(f"Energy forecast model failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        try:
            models['mood_classifier'] = self.build_mood_category_classifier(kaggle_train, listening_test)
            self.write_gold_table(models['mood_classifier']['predictions'],
                                'mood_classifications', gold_path)
        except Exception as e:
            logger.error(f"Mood classifier failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Save model metrics
        if models:
            self.save_model_metrics(models, gold_path)

        logger.info("=" * 80)
        logger.info("✅ PREDICTIVE ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Built {len(models)} predictive models")
        logger.info("Training: Kaggle dataset (114K tracks)")
        logger.info("Testing: User listening history (1,504 events)")


def main():
    """Main execution."""
    logger.info("Starting Predictive Analytics build...")

    # Paths
    bronze_path = '/app/data/bronze'
    silver_path = '/app/data/silver/listening_with_features'
    gold_path = '/app/data/gold'

    # Initialize Spark with Delta Lake support
    spark = SparkSession.builder \
        .appName("PredictiveAnalytics") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    try:
        analytics = PredictiveAnalytics(spark)
        analytics.build_all(bronze_path, silver_path, gold_path)

        logger.info("=" * 80)
        logger.info("✅ Predictive Analytics build SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Predictive Analytics build FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
