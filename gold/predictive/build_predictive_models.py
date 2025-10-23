"""
PREDICTIVE ANALYTICS - Gold Layer

Creates machine learning models to forecast future patterns:
1. Mood prediction based on time features
2. Energy level forecasting
3. Listening behavior prediction
4. Next track prediction

These tables answer: "What will happen?" and "What are future trends?"

Uses Spark MLlib for scalable ML.
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, hour, dayofweek
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

from utils.logger import setup_logger

logger = setup_logger('predictive_analytics')


class PredictiveAnalytics:
    """Build predictive ML models from Silver layer."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_silver_data(self, silver_path: str) -> DataFrame:
        """Load enriched listening data from Silver layer."""
        logger.info(f"Loading Silver layer data from: {silver_path}")
        try:
            df = self.spark.read.format('delta').load(silver_path)
            count = df.count()
            logger.info(f"✅ Loaded {count} records from Silver layer")
            return df
        except Exception as e:
            logger.error(f"Failed to load Silver data: {e}")
            raise

    def build_mood_prediction_model(self, df: DataFrame) -> dict:
        """
        PREDICTIVE: Predict valence (mood) based on time features.

        Model: Random Forest Regression
        Features: hour_of_day, day_of_week, is_weekend, energy, tempo
        Target: valence (happiness level)
        """
        logger.info("Building mood prediction model...")

        # Prepare features
        feature_cols = ['hour_of_day', 'day_of_week', 'energy', 'tempo', 'danceability']

        # Select columns including is_weekend if it exists
        select_cols = feature_cols + ['valence']
        if 'is_weekend' in df.columns:
            select_cols = select_cols + ['is_weekend']

        # Filter out nulls
        model_data = df.select(*select_cols) \
            .filter(col('valence').isNotNull())

        for feat in feature_cols:
            model_data = model_data.filter(col(feat).isNotNull())

        # Convert boolean to int if is_weekend exists
        if 'is_weekend' in df.columns:
            model_data = model_data.withColumn('is_weekend_int',
                                              when(col('is_weekend'), 1).otherwise(0))
            feature_cols_final = feature_cols + ['is_weekend_int']
        else:
            feature_cols_final = feature_cols

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols_final, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # Random Forest model (reduced complexity to prevent overfitting)
        rf = RandomForestRegressor(
            featuresCol='scaled_features',
            labelCol='valence',
            numTrees=20,  # Reduced from 50
            maxDepth=5,   # Reduced from 10
            minInstancesPerNode=5,  # Require at least 5 samples per leaf
            seed=42
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, rf])

        # Train/test split
        train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

        logger.info(f"Training on {train_data.count()} samples, testing on {test_data.count()} samples")

        # Train
        model = pipeline.fit(train_data)

        # Predict
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

        logger.info(f"✅ Mood Prediction Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'valence', 'prediction', 'energy', 'tempo'
        ).withColumnRenamed('prediction', 'predicted_valence')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'model_type': 'RandomForestRegressor'
        }

    def build_energy_forecast_model(self, df: DataFrame) -> dict:
        """
        PREDICTIVE: Forecast energy levels based on historical patterns.

        Model: Linear Regression
        Features: hour_of_day, previous_energy (lag), tempo
        Target: energy
        """
        logger.info("Building energy forecast model...")

        feature_cols = ['hour_of_day', 'day_of_week', 'tempo', 'danceability', 'loudness']

        # Filter out nulls
        model_data = df.select(*feature_cols, 'energy') \
            .filter(col('energy').isNotNull())

        for feat in feature_cols:
            model_data = model_data.filter(col(feat).isNotNull())

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

        # Train/test split
        train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

        logger.info(f"Training on {train_data.count()} samples, testing on {test_data.count()} samples")

        # Train
        model = pipeline.fit(train_data)

        # Predict
        predictions = model.transform(test_data)

        # Evaluate
        evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='rmse')
        rmse = evaluator.evaluate(predictions)
        mae_evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='mae')
        mae = mae_evaluator.evaluate(predictions)
        r2_evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='r2')
        r2 = r2_evaluator.evaluate(predictions)

        logger.info(f"✅ Energy Forecast Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'energy', 'prediction', 'tempo', 'danceability'
        ).withColumnRenamed('prediction', 'predicted_energy')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
            'model_type': 'LinearRegression'
        }

    def build_mood_category_classifier(self, df: DataFrame) -> dict:
        """
        PREDICTIVE: Classify listening sessions into mood categories.

        Model: Random Forest Classifier
        Features: hour_of_day, energy, tempo, acousticness
        Target: mood_category (Happy_Energetic, Happy_Calm, etc.)
        """
        logger.info("Building mood category classifier...")

        # Create mood categories based on valence and energy
        # NOTE: We EXCLUDE valence and energy from features to avoid data leakage
        mood_data = df.withColumn(
            'mood_label',
            when((col('valence') >= 0.6) & (col('energy') >= 0.6), 0)  # Happy_Energetic
            .when((col('valence') >= 0.6) & (col('energy') < 0.4), 1)   # Happy_Calm
            .when((col('valence') < 0.4) & (col('energy') >= 0.6), 2)   # Sad_Energetic
            .when((col('valence') < 0.4) & (col('energy') < 0.4), 3)    # Sad_Calm
            .otherwise(4)  # Neutral
        )

        # Features: Use only external/temporal features, NOT valence/energy (to avoid leakage)
        feature_cols = ['hour_of_day', 'day_of_week', 'tempo', 'danceability', 'acousticness', 'loudness', 'speechiness']

        # Filter nulls
        model_data = mood_data.select(*feature_cols, 'mood_label') \
            .filter(col('mood_label').isNotNull())

        for feat in feature_cols:
            model_data = model_data.filter(col(feat).isNotNull())

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

        # Random Forest Classifier (reduced complexity to prevent overfitting)
        rf_classifier = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol='mood_label',
            numTrees=20,  # Reduced from 50
            maxDepth=5,   # Reduced from 10
            minInstancesPerNode=5,  # Require at least 5 samples per leaf
            seed=42
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, rf_classifier])

        # Train/test split
        train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

        logger.info(f"Training on {train_data.count()} samples, testing on {test_data.count()} samples")

        # Train
        model = pipeline.fit(train_data)

        # Predict
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

        logger.info(f"✅ Mood Category Classifier - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'mood_label', 'prediction'
        ).withColumnRenamed('prediction', 'predicted_mood_label')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {'accuracy': accuracy, 'f1': f1},
            'model_type': 'RandomForestClassifier'
        }

    def write_gold_table(self, df: DataFrame, table_name: str, gold_path: str):
        """Write Gold layer table to Delta Lake."""
        output_path = f"{gold_path}/predictive/{table_name}"
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

    def save_model_metrics(self, models: dict, gold_path: str):
        """Save model performance metrics."""
        logger.info("Saving model metrics...")

        metrics_data = []
        for model_name, model_info in models.items():
            metrics = model_info['metrics']
            metrics_data.append({
                'model_name': model_name,
                'model_type': model_info['model_type'],
                **metrics
            })

        metrics_df = self.spark.createDataFrame(metrics_data)
        self.write_gold_table(metrics_df, 'model_performance_metrics', gold_path)

    def build_all(self, silver_path: str, gold_path: str):
        """Build all predictive analytics models."""
        logger.info("=" * 80)
        logger.info("BUILDING PREDICTIVE ANALYTICS - GOLD LAYER")
        logger.info("=" * 80)

        # Load Silver data
        silver_df = self.load_silver_data(silver_path)

        # Build all predictive models
        models = {}

        try:
            models['mood_prediction'] = self.build_mood_prediction_model(silver_df)
            self.write_gold_table(models['mood_prediction']['predictions'],
                                'mood_predictions', gold_path)
        except Exception as e:
            logger.error(f"Mood prediction model failed: {e}")

        try:
            models['energy_forecast'] = self.build_energy_forecast_model(silver_df)
            self.write_gold_table(models['energy_forecast']['predictions'],
                                'energy_forecasts', gold_path)
        except Exception as e:
            logger.error(f"Energy forecast model failed: {e}")

        try:
            models['mood_classifier'] = self.build_mood_category_classifier(silver_df)
            self.write_gold_table(models['mood_classifier']['predictions'],
                                'mood_classifications', gold_path)
        except Exception as e:
            logger.error(f"Mood classifier failed: {e}")

        # Save model metrics
        if models:
            self.save_model_metrics(models, gold_path)

        logger.info("=" * 80)
        logger.info("✅ PREDICTIVE ANALYTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Built {len(models)} predictive models")


def main():
    """Main execution."""
    logger.info("Starting Predictive Analytics build...")

    # Paths
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
        analytics.build_all(silver_path, gold_path)

        logger.info("=" * 80)
        logger.info("✅ Predictive Analytics build SUCCESSFUL")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Predictive Analytics build FAILED: {e}")
        return 1

    finally:
        spark.stop()


if __name__ == '__main__':
    sys.exit(main())
