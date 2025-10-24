"""
PREDICTIVE ANALYTICS - Gold Layer

⚠️ ACADEMIC DISCLAIMER:
This is a DEMONSTRATION of ML pipeline infrastructure, not scientifically valid predictions.

CRITICAL LIMITATIONS:
1. Test data uses SYNTHETIC audio features (Spotify API returns 403)
2. Train/Test distributions differ (Kaggle 2020 catalog vs. User 2024 listening)
3. Single-user dataset (n=1,504) prevents statistical generalization
4. Results demonstrate METHODOLOGY, not predictive accuracy

WHAT THIS DEMONSTRATES:
✅ ML pipeline architecture and model training workflows
✅ Cross-validation and proper model evaluation techniques
✅ Feature engineering and data quality handling
✅ Model persistence and prediction serving patterns

WHAT THIS DOES NOT PROVIDE:
❌ Scientifically valid music recommendations
❌ Statistically significant behavioral insights
❌ Production-ready predictive models

See ACADEMIC_DISCLAIMER.md for full details.

APPROACH:
- Train on Kaggle dataset (114K tracks with real audio features)
- Test on user's listening history (1,504 events with synthetic features)
- Use cross-validation on training data for honest performance estimates
- Compare against baseline models
- Provide feature importance analysis

Creates machine learning models to forecast future patterns:
1. Mood prediction based on time features
2. Energy level forecasting
3. Mood category classification

Uses Spark MLlib for scalable ML.
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, rand, abs as spark_abs, hour, dayofweek, lit, avg
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import numpy as np

from utils.logger import setup_logger

logger = setup_logger('predictive_analytics')


class PredictiveAnalytics:
    """Build predictive ML models trained on Kaggle, tested on real listening data."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def calculate_baseline_regression(self, test_data: DataFrame, label_col: str) -> dict:
        """
        Calculate baseline regression metrics (predict mean).
        This shows what a naive model would achieve.
        """
        logger.info(f"Calculating baseline (mean predictor) for {label_col}...")

        # Calculate mean of training label
        mean_value = test_data.select(avg(col(label_col))).collect()[0][0]

        # Add mean prediction
        baseline_predictions = test_data.withColumn('prediction', lit(mean_value))

        # Evaluate
        rmse_eval = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='rmse')
        mae_eval = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='mae')
        r2_eval = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='r2')

        baseline_rmse = rmse_eval.evaluate(baseline_predictions)
        baseline_mae = mae_eval.evaluate(baseline_predictions)
        baseline_r2 = r2_eval.evaluate(baseline_predictions)

        logger.info(f"   Baseline (Mean): RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}, R²={baseline_r2:.4f}")

        return {
            'rmse': baseline_rmse,
            'mae': baseline_mae,
            'r2': baseline_r2,
            'method': 'mean_predictor'
        }

    def calculate_baseline_classification(self, test_data: DataFrame, label_col: str) -> dict:
        """
        Calculate baseline classification metrics (predict most frequent class).
        """
        logger.info(f"Calculating baseline (majority class) for {label_col}...")

        # Find most frequent class
        class_counts = test_data.groupBy(label_col).count().collect()
        most_frequent_class = max(class_counts, key=lambda x: x['count'])[label_col]

        # Add majority class prediction
        baseline_predictions = test_data.withColumn('prediction', lit(most_frequent_class))

        # Evaluate
        accuracy_eval = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol='prediction', metricName='accuracy'
        )
        f1_eval = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol='prediction', metricName='f1'
        )

        baseline_accuracy = accuracy_eval.evaluate(baseline_predictions)
        baseline_f1 = f1_eval.evaluate(baseline_predictions)

        logger.info(f"   Baseline (Majority): Accuracy={baseline_accuracy:.4f}, F1={baseline_f1:.4f}")

        return {
            'accuracy': baseline_accuracy,
            'f1': baseline_f1,
            'method': 'majority_class'
        }

    def calculate_confusion_matrix(self, predictions: DataFrame, label_col: str = 'mood_label') -> dict:
        """
        Calculate confusion matrix for classification model.
        Returns class distribution and per-class metrics.
        """
        logger.info("Calculating confusion matrix and per-class metrics...")

        # Get predictions
        pred_labels = predictions.select(label_col, 'prediction').collect()

        # Count classes
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))

        for row in pred_labels:
            actual = int(row[label_col])
            predicted = int(row['prediction'])
            confusion[actual][predicted] += 1

        # Calculate per-class metrics
        classes = sorted(set([int(row[label_col]) for row in pred_labels]))
        per_class_metrics = {}

        mood_names = {
            0: 'Happy_Energetic',
            1: 'Happy_Calm',
            2: 'Sad_Energetic',
            3: 'Sad_Calm',
            4: 'Neutral'
        }

        logger.info("=" * 60)
        logger.info("CONFUSION MATRIX:")
        logger.info("=" * 60)

        for cls in classes:
            true_positive = confusion[cls][cls]
            false_positive = sum(confusion[other][cls] for other in classes if other != cls)
            false_negative = sum(confusion[cls][other] for other in classes if other != cls)

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_name = mood_names.get(cls, f'Class_{cls}')
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': true_positive + false_negative
            }

            logger.info(f"Class {cls} ({class_name}):")
            logger.info(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Support: {true_positive + false_negative}")

        logger.info("=" * 60)

        return {
            'confusion_matrix': dict(confusion),
            'per_class_metrics': per_class_metrics
        }

    def perform_cross_validation(self, train_data: DataFrame, pipeline: Pipeline,
                                 evaluator: RegressionEvaluator, num_folds: int = 3) -> dict:
        """
        Perform k-fold cross-validation on training data.
        This provides honest performance estimates on the training distribution.

        NOTE: We use 3 folds (not 5 or 10) due to dataset size and Spark overhead.
        """
        logger.info(f"Performing {num_folds}-fold cross-validation on training data...")
        logger.info("⚠️  This estimates performance on KAGGLE distribution, not user distribution")

        # Simple parameter grid (we're not tuning, just validating)
        paramGrid = ParamGridBuilder().build()

        # Cross-validator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42
        )

        # Fit with cross-validation
        cv_model = cv.fit(train_data)

        # Get average metric across folds
        avg_metric = float(np.mean(cv_model.avgMetrics))

        logger.info(f"   Cross-Validation Average {evaluator.getMetricName().upper()}: {avg_metric:.4f}")
        logger.info(f"   ✅ This represents expected performance on similar Kaggle-like data")

        return {
            'cv_metric': avg_metric,
            'metric_name': evaluator.getMetricName(),
            'num_folds': num_folds,
            'model': cv_model.bestModel
        }

    def extract_feature_importance(self, model, feature_cols: list) -> dict:
        """
        Extract feature importance from Random Forest models.
        Shows which features the model relies on most.
        """
        try:
            # Get the Random Forest model from the pipeline
            rf_model = None
            for stage in model.stages:
                if hasattr(stage, 'featureImportances'):
                    rf_model = stage
                    break

            if rf_model is None:
                logger.warning("Could not extract feature importances (not a tree-based model)")
                return {}

            importances = rf_model.featureImportances.toArray()

            feature_importance = {
                feature_cols[i]: float(importances[i])
                for i in range(len(feature_cols))
            }

            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            logger.info("=" * 60)
            logger.info("FEATURE IMPORTANCE:")
            logger.info("=" * 60)
            for feat, importance in sorted_features:
                logger.info(f"  {feat:20s}: {importance:.4f}")
            logger.info("=" * 60)

            return feature_importance

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}

    def load_kaggle_training_data(self, bronze_path: str) -> DataFrame:
        """Load Kaggle dataset as TRAINING data (114K tracks with real features)."""
        logger.info(f"Loading Kaggle training data from: {bronze_path}")
        try:
            df = self.spark.read.format('delta').load(f"{bronze_path}/kaggle_tracks_bronze")
            count_before = df.count()
            logger.info(f"✅ Loaded {count_before} Kaggle tracks (raw)")

            # DATA QUALITY FIX: Remove corrupted rows with out-of-range values
            # Audio features should be in range [0, 1]
            logger.info("Cleaning data: removing corrupted rows...")
            df = df.filter(
                (col('valence').between(0, 1)) &
                (col('energy').between(0, 1)) &
                (col('danceability').between(0, 1)) &
                (col('acousticness').between(0, 1)) &
                (col('instrumentalness').between(0, 1)) &
                (col('tempo').between(40, 250))  # Realistic BPM range
            )

            count_after = df.count()
            removed = count_before - count_after
            logger.info(f"✅ Cleaned data: {count_after} valid rows ({removed} corrupted rows removed)")

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

        # STEP 1: Calculate baseline on test data
        logger.info("")
        logger.info("STEP 1: Baseline Model (Mean Predictor)")
        baseline_metrics = self.calculate_baseline_regression(test_data, 'valence')

        # STEP 2: Cross-validate on training data (Kaggle)
        logger.info("")
        logger.info("STEP 2: Cross-Validation on Training Data (Kaggle)")
        evaluator = RegressionEvaluator(
            labelCol='valence',
            predictionCol='prediction',
            metricName='rmse'
        )
        cv_results = self.perform_cross_validation(train_data, pipeline, evaluator, num_folds=3)

        # Use the cross-validated model
        model = cv_results['model']

        # STEP 3: Test on user's listening history
        logger.info("")
        logger.info("STEP 3: Testing on User's Listening Data (Synthetic Features)")
        predictions = model.transform(test_data)

        # Evaluate on test data
        rmse = evaluator.evaluate(predictions)
        mae_evaluator = RegressionEvaluator(labelCol='valence', predictionCol='prediction', metricName='mae')
        mae = mae_evaluator.evaluate(predictions)
        r2_evaluator = RegressionEvaluator(labelCol='valence', predictionCol='prediction', metricName='r2')
        r2 = r2_evaluator.evaluate(predictions)

        # STEP 4: Extract feature importance
        logger.info("")
        logger.info("STEP 4: Feature Importance Analysis")
        feature_importance = self.extract_feature_importance(model, feature_cols_final)

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ MOOD PREDICTION MODEL RESULTS")
        logger.info("=" * 80)
        logger.info("Training Data: Kaggle (114K tracks, real features)")
        logger.info(f"  Cross-Val RMSE: {cv_results['cv_metric']:.4f} (on Kaggle distribution)")
        logger.info("")
        logger.info("Test Data: User Listening (1.5K events, SYNTHETIC features)")
        logger.info(f"  Baseline RMSE:  {baseline_metrics['rmse']:.4f} (mean predictor)")
        logger.info(f"  Model RMSE:     {rmse:.4f}")
        logger.info(f"  Model MAE:      {mae:.4f}")
        logger.info(f"  Model R²:       {r2:.4f}")
        logger.info("")
        if r2 < 0:
            logger.warning("⚠️  NEGATIVE R²: Model performs worse than baseline on test distribution")
            logger.warning("   This is EXPECTED due to train/test distribution mismatch + synthetic features")
        logger.info("=" * 80)

        # Save predictions on user's listening data
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'valence', 'prediction', 'energy', 'tempo'
        ).withColumnRenamed('prediction', 'predicted_valence')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'cv_rmse': cv_results['cv_metric'],
                'baseline_rmse': baseline_metrics['rmse'],
                'baseline_r2': baseline_metrics['r2'],
                'train_size': train_count,
                'test_size': test_count
            },
            'feature_importance': feature_importance,
            'model_type': 'RandomForestRegressor',
            'training_approach': 'Kaggle_to_User_with_CrossVal'
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

        # STEP 1: Calculate baseline on test data
        logger.info("")
        logger.info("STEP 1: Baseline Model (Mean Predictor)")
        baseline_metrics = self.calculate_baseline_regression(test_data, 'energy')

        # STEP 2: Cross-validate on training data (Kaggle)
        logger.info("")
        logger.info("STEP 2: Cross-Validation on Training Data (Kaggle)")
        evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='rmse')
        cv_results = self.perform_cross_validation(train_data, pipeline, evaluator, num_folds=3)

        # Use the cross-validated model
        model = cv_results['model']

        # STEP 3: Test on user's listening history
        logger.info("")
        logger.info("STEP 3: Testing on User's Listening Data (Synthetic Features)")
        predictions = model.transform(test_data)

        # Evaluate
        rmse = evaluator.evaluate(predictions)
        mae_evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='mae')
        mae = mae_evaluator.evaluate(predictions)
        r2_evaluator = RegressionEvaluator(labelCol='energy', predictionCol='prediction', metricName='r2')
        r2 = r2_evaluator.evaluate(predictions)

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ ENERGY FORECAST MODEL RESULTS")
        logger.info("=" * 80)
        logger.info("Training Data: Kaggle (114K tracks, real features)")
        logger.info(f"  Cross-Val RMSE: {cv_results['cv_metric']:.4f} (on Kaggle distribution)")
        logger.info("")
        logger.info("Test Data: User Listening (1.5K events, SYNTHETIC features)")
        logger.info(f"  Baseline RMSE:  {baseline_metrics['rmse']:.4f} (mean predictor)")
        logger.info(f"  Model RMSE:     {rmse:.4f}")
        logger.info(f"  Model MAE:      {mae:.4f}")
        logger.info(f"  Model R²:       {r2:.4f}")
        logger.info("")
        if r2 < 0:
            logger.warning("⚠️  NEGATIVE R²: Model performs worse than baseline on test distribution")
            logger.warning("   This is EXPECTED due to train/test distribution mismatch + synthetic features")
        logger.info("=" * 80)

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'energy', 'prediction', 'tempo', 'danceability'
        ).withColumnRenamed('prediction', 'predicted_energy')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'cv_rmse': cv_results['cv_metric'],
                'baseline_rmse': baseline_metrics['rmse'],
                'baseline_r2': baseline_metrics['r2'],
                'train_size': train_count,
                'test_size': test_count
            },
            'model_type': 'LinearRegression',
            'training_approach': 'Kaggle_to_User_with_CrossVal'
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

        # STEP 1: Calculate baseline on test data
        logger.info("")
        logger.info("STEP 1: Baseline Model (Majority Class)")
        baseline_metrics = self.calculate_baseline_classification(test_data, 'mood_label')

        # STEP 2: Cross-validate on training data (Kaggle)
        logger.info("")
        logger.info("STEP 2: Cross-Validation on Training Data (Kaggle)")
        evaluator = MulticlassClassificationEvaluator(
            labelCol='mood_label',
            predictionCol='prediction',
            metricName='accuracy'
        )
        cv_results = self.perform_cross_validation(train_data, pipeline, evaluator, num_folds=3)

        # Use the cross-validated model
        model = cv_results['model']

        # STEP 3: Test on user's listening history
        logger.info("")
        logger.info("STEP 3: Testing on User's Listening Data (Synthetic Features)")
        predictions = model.transform(test_data)

        # Evaluate
        accuracy = evaluator.evaluate(predictions)
        f1_evaluator = MulticlassClassificationEvaluator(labelCol='mood_label', predictionCol='prediction', metricName='f1')
        f1 = f1_evaluator.evaluate(predictions)

        # STEP 4: Calculate confusion matrix and per-class metrics
        logger.info("")
        logger.info("STEP 4: Confusion Matrix & Per-Class Analysis")
        confusion_results = self.calculate_confusion_matrix(predictions, 'mood_label')

        # STEP 5: Extract feature importance
        logger.info("")
        logger.info("STEP 5: Feature Importance Analysis")
        feature_importance = self.extract_feature_importance(model, feature_cols)

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ MOOD CATEGORY CLASSIFIER RESULTS")
        logger.info("=" * 80)
        logger.info("Training Data: Kaggle (114K tracks, real features)")
        logger.info(f"  Cross-Val Accuracy: {cv_results['cv_metric']:.4f} (on Kaggle distribution)")
        logger.info("")
        logger.info("Test Data: User Listening (1.5K events, SYNTHETIC features)")
        logger.info(f"  Baseline Accuracy:  {baseline_metrics['accuracy']:.4f} (majority class)")
        logger.info(f"  Model Accuracy:     {accuracy:.4f}")
        logger.info(f"  Model F1:           {f1:.4f}")
        logger.info("")
        logger.warning("⚠️  Classification on SYNTHETIC features - results demonstrate pipeline only")
        logger.info("=" * 80)

        # Save predictions
        prediction_results = predictions.select(
            'hour_of_day', 'day_of_week', 'mood_label', 'prediction'
        ).withColumnRenamed('prediction', 'predicted_mood_label')

        return {
            'model': model,
            'predictions': prediction_results,
            'metrics': {
                'test_accuracy': accuracy,
                'test_f1': f1,
                'cv_accuracy': cv_results['cv_metric'],
                'baseline_accuracy': baseline_metrics['accuracy'],
                'baseline_f1': baseline_metrics['f1'],
                'train_size': train_count,
                'test_size': test_count
            },
            'confusion_matrix': confusion_results,
            'feature_importance': feature_importance,
            'model_type': 'RandomForestClassifier',
            'training_approach': 'Kaggle_to_User_with_CrossVal'
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
