# Spotify Analytics Pipeline - Manual Execution Guide

**Purpose:** Step-by-step manual execution and verification of the complete pipeline

**Environment:** Dockerized (docker-compose)

**Date:** 2025-10-24

---

## 📋 Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Stage 1: Bronze Layer Ingestion](#stage-1-bronze-layer-ingestion)
3. [Stage 2: Silver Layer Transformation](#stage-2-silver-layer-transformation)
4. [Stage 3: Gold Layer Analytics](#stage-3-gold-layer-analytics)
5. [Stage 4: ML Model Training](#stage-4-ml-model-training)
6. [Trino/Presto Testing](#trinopresto-testing)
7. [Superset Dashboard Refresh](#superset-dashboard-refresh)
8. [ML Model Improvements](#ml-model-improvements)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Setup

### 1. Start All Services

```bash
# Start all Docker containers
docker-compose up -d

# Verify all services are running
docker ps

# Expected containers:
# - spotify-scheduler (for automated runs - can ignore for manual)
# - trino
# - superset
# - postgres (Superset metadata)
# - redis (Superset cache)
```

### 2. Verify Environment

```bash
# Check Trino is healthy
docker logs trino --tail 20

# Check Superset is ready
docker logs superset --tail 20

# Verify data directories exist
ls -la data/bronze/
ls -la data/silver/
ls -la data/gold/
```

### 3. Check Kaggle Dataset (ONE-TIME SETUP)

```bash
# Verify Kaggle CSV exists (114K tracks, ~40MB)
ls -lh data/kaggle/dataset.csv

# If missing, download from Kaggle:
# https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
# Place in: data/kaggle/dataset.csv
```

---

## Stage 1: Bronze Layer Ingestion

### 1.1 Ingest Kaggle Dataset (ONE-TIME ONLY)

**Purpose:** Load 114K Spotify tracks with real audio features into Bronze layer

**When to run:** ONLY ONCE, or when you want to refresh the Kaggle dataset

**Command:**
```bash
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

**What it does:**
- Reads `data/kaggle/dataset.csv` (114K rows)
- Maps to Bronze schema
- Writes to `data/bronze/kaggle_tracks_bronze/` (Delta table)
- **Mode:** OVERWRITE (replaces existing data)

**Verification:**
```bash
# Check Bronze Kaggle table was created
ls -la data/bronze/kaggle_tracks_bronze/

# Count rows (should be ~113,692 after cleaning)
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()
df = spark.read.format('delta').load('/app/data/bronze/kaggle_tracks_bronze')
print(f'Kaggle tracks: {df.count()} rows')
df.show(5)
spark.stop()
"
```

**Expected Output:**
```
Kaggle tracks: 113692 rows
+--------------------+--------------------+-------+------+------------+
|          track_name|             artists|valence|energy|danceability|
+--------------------+--------------------+-------+------+------------+
|Te Enseñare A Olv...|        Grupo Niche|  0.905| 0.752|       0.665|
|       Ole con Ole  |      Los Master's  |  0.966| 0.471|       0.715|
+--------------------+--------------------+-------+------+------------+
```

**Note:** Kaggle ingestion happens automatically as part of `run_ingestion.py` but only overwrites, not appends. For subsequent runs, it will reload the same data (idempotent).

---

### 1.2 Ingest Spotify API Data (INCREMENTAL)

**Purpose:** Fetch your recent listening history from Spotify API

**When to run:** Every time you want fresh listening data (hourly, daily, etc.)

**Command:**
```bash
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

**What it does:**
- Fetches recently played tracks (up to 1000 items)
- Fetches track metadata
- Attempts to fetch audio features (may fail with 403 if no permissions)
- Falls back to synthetic feature generation
- Writes to:
  - `data/bronze/listening_history_bronze/` (Delta table, APPEND mode)
  - `data/bronze/my_tracks_features_bronze/` (Delta table, APPEND mode)
  - `data/bronze/my_tracks_features_bronze_synthetic/` (Delta table, APPEND mode)

**Verification:**
```bash
# Check listening history table
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

print('=== LISTENING HISTORY ===')
df = spark.read.format('delta').load('/app/data/bronze/listening_history_bronze')
print(f'Total listens: {df.count()}')
df.orderBy('played_at', ascending=False).show(10, truncate=False)

print('\n=== SYNTHETIC FEATURES ===')
df2 = spark.read.format('delta').load('/app/data/bronze/my_tracks_features_bronze_synthetic')
print(f'Total tracks with features: {df2.count()}')
df2.show(5)

spark.stop()
"
```

**Expected Output:**
```
=== LISTENING HISTORY ===
Total listens: 1504
+--------------------+--------------------+-------------------+
|          track_name|         artist_name|          played_at|
+--------------------+--------------------+-------------------+
|         Song Title |        Artist Name | 2025-10-24 12:30:00|
+--------------------+--------------------+-------------------+

=== SYNTHETIC FEATURES ===
Total tracks with features: 1504
+--------------------+-------+------+------------+
|          track_name|valence|energy|danceability|
+--------------------+-------+------+------------+
|         Song Title | 0.654 | 0.512|       0.701|
+--------------------+-------+------+------------+
```

**Important Notes:**
- `run_ingestion.py` runs ALL Bronze ingestion (Kaggle + Spotify API + Synthetic)
- Kaggle overwrites (idempotent), Spotify appends (incremental)
- Synthetic features are auto-generated for any tracks missing audio features

---

## Stage 2: Silver Layer Transformation

### 2.1 Build Silver: Listening with Features

**Purpose:** Join listening history with audio features (real or synthetic)

**Command:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py
```

**What it does:**
- Reads Bronze: `listening_history_bronze` (1,504 listens)
- Reads Bronze: `my_tracks_features_bronze_synthetic` (1,504 features)
- Joins listening events with audio features
- Enriches with time features (hour, day_of_week, part_of_day)
- Writes to `data/silver/listening_with_features/` (Delta table, OVERWRITE mode)

**Verification:**
```bash
# Check Silver table
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

df = spark.read.format('delta').load('/app/data/silver/listening_with_features')
print(f'Silver records: {df.count()}')
print(f'Columns: {len(df.columns)}')
print('\nSchema:')
df.printSchema()
print('\nSample data:')
df.select('track_name', 'artist_name', 'valence', 'energy', 'hour_of_day', 'part_of_day').show(10)

# Check for nulls
print('\nData quality check:')
print(f\"Valence nulls: {df.filter(df.valence.isNull()).count()}\")
print(f\"Energy nulls: {df.filter(df.energy.isNull()).count()}\")

spark.stop()
"
```

**Expected Output:**
```
Silver records: 1504
Columns: 25+

Schema:
 |-- track_id: string
 |-- track_name: string
 |-- artist_name: string
 |-- played_at: timestamp
 |-- valence: float
 |-- energy: float
 |-- tempo: float
 |-- danceability: float
 |-- acousticness: float
 |-- hour_of_day: integer
 |-- day_of_week: integer
 |-- part_of_day: string
 |-- is_weekend: boolean
 ...

Sample data:
+--------------------+--------------------+-------+------+-----------+------------+
|          track_name|         artist_name|valence|energy|hour_of_day|part_of_day |
+--------------------+--------------------+-------+------+-----------+------------+
|         Song Title |             Artist| 0.654 | 0.512|         14|afternoon   |
+--------------------+--------------------+-------+------+-----------+------------+

Data quality check:
Valence nulls: 0
Energy nulls: 0
```

---

## Stage 3: Gold Layer Analytics

### 3.1 Descriptive Analytics (What happened?)

**Purpose:** Aggregate statistics about listening patterns

**Command:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 gold/descriptive/build_descriptive_analytics.py
```

**What it creates:**
- `listening_patterns_by_time` - Hourly/daily listening patterns
- `top_tracks_by_mood` - Most played tracks by mood category
- `temporal_trends` - Time-based trends
- `audio_feature_distributions` - Feature distributions
- `feature_source_coverage` - Data quality metrics

**Verification:**
```bash
# Check all descriptive tables exist
ls -la data/gold/descriptive/

# Query a sample table
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

df = spark.read.format('delta').load('/app/data/gold/descriptive/listening_patterns_by_time')
print(f'Patterns: {df.count()} rows')
df.orderBy('hour_of_day').show(10)

spark.stop()
"
```

---

### 3.2 Diagnostic Analytics (Why did it happen?)

**Purpose:** Analyze correlations and causal relationships

**Command:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 gold/diagnostic/build_diagnostic_analytics.py
```

**What it creates:**
- `mood_time_correlations` - Mood vs time correlations
- `feature_correlations` - Audio feature correlations
- `weekend_vs_weekday` - Weekend vs weekday comparisons
- `mood_shift_patterns` - Mood change patterns
- `part_of_day_drivers` - What drives listening by time of day

**Verification:**
```bash
# Check correlations
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

df = spark.read.format('delta').load('/app/data/gold/diagnostic/mood_time_correlations')
print('Mood-Time Correlations:')
df.show()

spark.stop()
"
```

---

### 3.3 Prescriptive Analytics (What should we do?)

**Purpose:** Generate recommendations and interventions

**Command:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 gold/prescriptive/build_prescriptive_analytics.py
```

**What it creates:**
- `mood_improvement_recommendations` - Tracks that boost mood
- `optimal_listening_times` - Best times to listen
- `personalized_playlist_suggestions` - Auto-generated playlists
- `mood_intervention_triggers` - When to intervene

**Verification:**
```bash
# Check recommendations
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

df = spark.read.format('delta').load('/app/data/gold/prescriptive/mood_improvement_recommendations')
print('Mood Improvement Recommendations:')
df.show(10, truncate=False)

spark.stop()
"
```

---

### 3.4 Cognitive Analytics (Complex patterns)

**Purpose:** Discover hidden patterns using clustering and anomaly detection

**Command:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 gold/cognitive/build_cognitive_analytics.py
```

**What it creates:**
- `mood_state_clusters` - K-means mood clusters
- `listening_anomalies` - Unusual patterns
- `sequential_patterns` - Mood sequences
- `behavioral_segments` - User archetypes

**Verification:**
```bash
# Check clusters
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

df = spark.read.format('delta').load('/app/data/gold/cognitive/mood_state_clusters')
print('Mood Clusters:')
df.show(truncate=False)

spark.stop()
"
```

---

## Stage 4: ML Model Training

### 4.1 Predictive Analytics + ML Models (What will happen?)

**Purpose:** Train machine learning models for mood prediction

**Command:**
```bash
docker-compose run --rm spotify-pipeline bash -c \
  "pip3 install --no-cache-dir -q numpy==1.24.4 scikit-learn==1.3.2 pandas==2.0.3 && \
   python3 gold/predictive/build_predictive_models.py"
```

**What it does:**
- Loads Kaggle dataset (113,692 tracks) as TRAINING data
- Loads your listening history (1,504 events) as TEST data
- Trains 3 models:
  1. **Mood Prediction** (Random Forest Regressor)
  2. **Energy Forecast** (Linear Regression)
  3. **Mood Category Classifier** (Random Forest Classifier)
- Evaluates on test set
- Saves predictions and metrics

**Output Tables:**
- `mood_predictions` - Predicted valence values
- `energy_forecasts` - Predicted energy values
- `mood_classifications` - Predicted mood categories
- `model_performance_metrics` - Model evaluation metrics

**Verification:**
```bash
# Check model performance
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

print('=== MODEL PERFORMANCE METRICS ===')
df = spark.read.format('delta').load('/app/data/gold/predictive/model_performance_metrics')
df.show(truncate=False)

print('\n=== MOOD PREDICTIONS SAMPLE ===')
df2 = spark.read.format('delta').load('/app/data/gold/predictive/mood_predictions')
df2.select('hour_of_day', 'valence', 'predicted_valence', 'energy').show(10)

spark.stop()
"
```

**Expected Output:**
```
=== MODEL PERFORMANCE METRICS ===
+------------------+---------------------+--------------------+----------+---------+------+----------+---------+
|model_name        |model_type           |training_approach   |train_size|test_size|rmse  |mae       |r2       |
+------------------+---------------------+--------------------+----------+---------+------+----------+---------+
|mood_prediction   |RandomForestRegressor|Kaggle_to_User      |113692    |1504     |0.1725|-0.1344   |-0.2848  |
|energy_forecast   |LinearRegression     |Kaggle_to_User      |113692    |1504     |0.2481|0.1970    |-0.5712  |
|mood_classifier   |RandomForestClassifier|Kaggle_to_User     |113692    |1504     |null  |null      |null     |
+------------------+---------------------+--------------------+----------+---------+------+----------+---------+

Mood Classifier Accuracy: 56.98%, F1: 0.5094
```

**Note:** Negative R² indicates the model performs worse than simply predicting the mean. This is expected when testing on a different distribution (general Kaggle music → your specific taste).

---

## Trino/Presto Testing

### 5.1 Connect to Trino CLI

**Interactive SQL Shell:**
```bash
# Connect to Trino CLI
docker exec -it trino trino --catalog delta --schema default

# You'll see:
# trino:default>
```

**Example queries:**
```sql
-- List all tables
SHOW TABLES;

-- Count rows in Silver layer
SELECT COUNT(*) FROM listening_with_features;

-- Top artists
SELECT artist_name, COUNT(*) as plays
FROM listening_with_features
GROUP BY artist_name
ORDER BY plays DESC
LIMIT 10;

-- Average mood by hour
SELECT hour_of_day, AVG(valence) as avg_mood
FROM listening_with_features
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- Exit
quit;
```

---

### 5.2 Connect to Trino from Python

**Create test script:**
```bash
docker-compose run --rm spotify-pipeline python3 -c "
from trino.dbapi import connect
from trino.auth import BasicAuthentication

# Connect to Trino
conn = connect(
    host='trino',
    port=8080,
    catalog='delta',
    schema='default',
    user='admin'
)

cursor = conn.cursor()

# Query tables
cursor.execute('SHOW TABLES')
tables = cursor.fetchall()
print('Available tables:')
for table in tables:
    print(f'  - {table[0]}')

# Query data
cursor.execute('SELECT COUNT(*) FROM listening_with_features')
count = cursor.fetchone()[0]
print(f'\nListening events: {count}')

cursor.close()
conn.close()
"
```

**Note:** You need to install `trino` package first:
```bash
docker exec superset pip install trino
```

---

### 5.3 Verify Data Updated

**Check last ingestion time:**
```bash
docker exec -it trino trino --catalog delta --schema default --execute \
  "SELECT MAX(played_at) as last_play,
          MIN(played_at) as first_play,
          COUNT(*) as total_plays
   FROM listening_with_features"
```

**Check table file modification time:**
```bash
ls -lt data/silver/listening_with_features/ | head -10
```

**Verify new data in Gold:**
```bash
docker exec -it trino trino --catalog delta --schema default --execute \
  "SELECT COUNT(*) FROM mood_predictions"
```

---

### 5.4 Trino Web UI

**Access:** http://localhost:8080

- View running queries
- Check cluster stats
- Monitor query performance

---

## Superset Dashboard Refresh

### 6.1 Access Superset

**URL:** http://localhost:8088

**Login:**
- Username: `admin`
- Password: `admin`

---

### 6.2 Connect Superset to Trino

1. Go to **Settings** → **Database Connections**
2. Click **+ Database**
3. Select **Trino**
4. Enter connection details:
   ```
   Display Name: Spotify Analytics (Trino)
   SQLAlchemy URI: trino://admin@trino:8080/delta/default
   ```
5. Click **Test Connection** → Should see ✅
6. Click **Connect**

---

### 6.3 Create Datasets from Tables

1. Go to **Data** → **Datasets**
2. Click **+ Dataset**
3. Select:
   - Database: `Spotify Analytics (Trino)`
   - Schema: `default`
   - Table: `listening_with_features` (or any Gold table)
4. Click **Add**

Repeat for all 22 Gold tables you want to visualize.

---

### 6.4 Create Charts

1. Go to **Charts**
2. Click **+ Chart**
3. Select:
   - Dataset: `listening_with_features`
   - Chart Type: `Line Chart` (or any type)
4. Configure:
   - X-Axis: `hour_of_day`
   - Metric: `COUNT(*)`
5. Click **Create Chart**
6. Customize and save

---

### 6.5 Refresh Data in Superset

**Option A: Automatic (via Trino)**
- Charts refresh automatically when queried
- Trino queries Delta Lake directly (always fresh)

**Option B: Manual Refresh**
- Open a dashboard
- Click **Refresh** button in top right

**Option C: Scheduled Refresh**
- Edit chart
- Go to **Advanced** → **Cache Timeout**
- Set cache duration (e.g., 3600 seconds = 1 hour)

---

## ML Model Improvements

### 7.1 Current Model Architecture

**Models:**
1. **Mood Prediction** - Random Forest Regressor
   - Features: hour, day, energy, tempo, danceability
   - Target: valence (happiness)
   - Performance: R² = -0.28 (poor generalization)

2. **Energy Forecast** - Linear Regression
   - Features: hour, day, tempo, danceability, acousticness
   - Target: energy
   - Performance: R² = -0.57 (poor generalization)

3. **Mood Classifier** - Random Forest Classifier
   - Features: hour, day, tempo, danceability, acousticness, instrumentalness
   - Target: mood category (5 classes)
   - Performance: 57% accuracy (moderate)

**Problem:** Models train on general Kaggle music (114K tracks) but test on specific user taste (1.5K events) → poor generalization

---

### 7.2 Suggested Improvements

#### Improvement 1: Use User Data for Training (NOT Kaggle)

**Current Approach:**
```
Train: Kaggle (114K general music)
Test: User (1.5K specific taste)
Problem: Distribution mismatch
```

**Better Approach:**
```
Train/Test: User data ONLY (1.5K events)
Split: 80% train (1,203), 20% test (301)
Benefit: Same distribution
```

**Implementation:**
```python
# In build_predictive_models.py, change:
# train_df = load_kaggle_training_data()  ❌
# test_df = load_listening_test_data()

# To:
df = load_listening_test_data()  # Load user data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)  # Split user data
```

---

#### Improvement 2: Add Temporal Features

**Current Features:** hour_of_day, day_of_week (basic)

**Add:**
- **Cyclical encoding** (sin/cos for hour/day)
  ```python
  import math
  df = df.withColumn('hour_sin', sin(2 * math.pi * col('hour_of_day') / 24))
  df = df.withColumn('hour_cos', cos(2 * math.pi * col('hour_of_day') / 24))
  df = df.withColumn('day_sin', sin(2 * math.pi * col('day_of_week') / 7))
  df = df.withColumn('day_cos', cos(2 * math.pi * col('day_of_week') / 7))
  ```

- **Time since last listen** (sequence feature)
- **Rolling averages** (last 5, 10, 20 tracks)

---

#### Improvement 3: Use Sequence Models (LSTM/GRU)

**Current:** Each listen is independent

**Better:** Model listening sequences

**Implementation:**
```python
# Convert to sequences
# Input: [track1, track2, track3, ...]
# Predict: next track's mood

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(sequence_length, n_features)),
    Dense(32, activation='relu'),
    Dense(1)  # Predict valence
])
```

---

#### Improvement 4: Transfer Learning with Audio Embeddings

**Problem:** Limited data (1.5K tracks)

**Solution:** Use pre-trained audio embeddings

**Options:**
1. **VGGish** (Google's audio CNN)
2. **OpenL3** (Music + environmental audio)
3. **CLMR** (Contrastive learning for music)

**Implementation:**
```python
# Extract embeddings using VGGish
import tensorflow as tf
import tensorflow_hub as hub

# Load pre-trained model
vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

# Extract embeddings from audio files
# (Requires actual .mp3 files, not just features)
embeddings = vggish_model(audio_samples)

# Use embeddings as features
features = np.concatenate([time_features, embeddings], axis=1)
```

**Challenge:** Requires audio files, not just Spotify API features

---

#### Improvement 5: Gaussian Mixture Models (GMM) for Clustering

**Current:** K-Means (hard clustering, assumes spherical clusters)

**Better:** GMM (soft clustering, handles complex shapes)

**Implementation:**
```python
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(
    featuresCol='scaled_features',
    k=5,  # 5 mood states
    seed=42
)

model = gmm.fit(train_data)
predictions = model.transform(test_data)
```

**Benefits:**
- Soft clustering (probabilistic membership)
- Better handles overlapping mood states
- More robust with limited data

---

#### Improvement 6: Autoencoders for Dimensionality Reduction

**Purpose:** Learn compressed representations of listening patterns

**Implementation:**
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Encoder
input_layer = Input(shape=(n_features,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(n_features, activation='sigmoid')(decoded)

# Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(features, features, epochs=50)

# Use encoded representation for clustering
encoder = Model(input_layer, encoded)
compressed_features = encoder.predict(features)
```

---

#### Improvement 7: Hyperparameter Tuning

**Current:** Default parameters

**Better:** Grid search / Random search

**Implementation:**
```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Define parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30, 50]) \
    .addGrid(rf.maxDepth, [5, 10, 15, 20]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Cross-validation
cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(metricName='rmse'),
    numFolds=5
)

# Fit
cvModel = cv.fit(train_data)
best_model = cvModel.bestModel
```

---

### 7.3 Manual Model Retraining Commands

**Retrain with improvements:**
```bash
# 1. Edit build_predictive_models.py with improvements

# 2. Retrain models
docker-compose run --rm spotify-pipeline bash -c \
  "pip3 install --no-cache-dir -q numpy scikit-learn pandas tensorflow && \
   python3 gold/predictive/build_predictive_models.py"

# 3. Evaluate new models
docker-compose run --rm spotify-pipeline python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.jars.packages', 'io.delta:delta-spark_2.12:3.2.1') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .getOrCreate()

df = spark.read.format('delta').load('/app/data/gold/predictive/model_performance_metrics')
df.show(truncate=False)
spark.stop()
"

# 4. Compare old vs new metrics
# Save old metrics before retraining:
docker cp spotify-scheduler:/app/data/gold/predictive/model_performance_metrics/ \
  ./model_metrics_backup/
```

---

## Kaggle Dataset Handling

### 8.1 Current Behavior

**Verification:**
```bash
# Check run_ingestion.py code
docker-compose run --rm spotify-pipeline python3 -c "
import subprocess
result = subprocess.run(['grep', '-n', 'kaggle', 'run_ingestion.py'],
                       capture_output=True, text=True)
print(result.stdout)
"
```

**What happens:**
- `run_ingestion.py` calls `ingest_kaggle_dataset()`
- Mode: **OVERWRITE** (not append)
- Kaggle CSV is read every time
- Delta table is overwritten with same data

**Is this a problem?**
- ❌ Wastes time (reads 114K rows unnecessarily)
- ❌ Wastes CPU (re-parses same CSV)
- ✅ BUT: Ensures data is always clean
- ✅ Idempotent (running twice = same result)

---

### 8.2 Optimal Behavior

**Goal:**
- Load Kaggle ONCE (initial setup)
- Skip Kaggle on subsequent runs
- Only update Spotify API data

**Implementation:**
```python
# In run_ingestion.py

def ingest_kaggle_dataset(spark: SparkSession, config) -> None:
    """Load Kaggle dataset ONLY IF not already loaded."""

    # Check if Kaggle Bronze table exists
    kaggle_path = Path(config.paths.kaggle_tracks)

    if kaggle_path.exists():
        logger.info("Kaggle dataset already loaded. Skipping...")

        # Optionally verify row count
        df = spark.read.format('delta').load(str(kaggle_path))
        count = df.count()
        logger.info(f"Existing Kaggle dataset: {count:,} rows")
        return

    # Only load if table doesn't exist
    logger.info("Kaggle dataset not found. Loading for first time...")
    # ... existing loading code ...
```

**To apply this fix:**
```bash
# Edit run_ingestion.py manually or use the code above

# Test: First run (loads Kaggle)
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
# Logs: "Loading Kaggle dataset..."

# Test: Second run (skips Kaggle)
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
# Logs: "Kaggle dataset already loaded. Skipping..."
```

---

### 8.3 Force Kaggle Reload

**When needed:**
- Updated Kaggle CSV file
- Found data quality issues
- Want to re-clean data

**Command:**
```bash
# Delete Kaggle Bronze table
rm -rf data/bronze/kaggle_tracks_bronze/

# Re-run ingestion
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
# Will reload Kaggle from scratch
```

---

## Troubleshooting

### Common Issues

**Issue 1: Spotify API 403 errors**
```
Solution: Your Spotify app needs additional scopes
- Go to: https://developer.spotify.com/dashboard
- Edit your app
- Add scope: user-read-recently-played
- Regenerate tokens
```

**Issue 2: Delta Lake tables not found**
```bash
# Rebuild tables
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

**Issue 3: Trino can't find tables**
```bash
# Restart Trino
docker-compose restart trino

# Check catalog config
docker exec trino cat /etc/trino/catalog/delta.properties
```

**Issue 4: Superset connection fails**
```bash
# Install Trino driver in Superset
docker exec superset pip install trino sqlalchemy-trino

# Restart Superset
docker-compose restart superset
```

**Issue 5: ML models fail to train**
```bash
# Install ML libraries
docker exec spotify-scheduler pip install numpy scikit-learn pandas

# Re-run training
docker-compose run --rm spotify-pipeline python3 gold/predictive/build_predictive_models.py
```

---

## Quick Reference: Complete Pipeline Execution

**Run entire pipeline manually (copy-paste all):**

```bash
# 1. Bronze: Ingest all data (Kaggle + Spotify API)
docker-compose run --rm spotify-pipeline python3 run_ingestion.py

# 2. Silver: Transform and enrich
docker-compose run --rm spotify-pipeline python3 scripts/build_silver_listening_with_features.py

# 3. Gold: Descriptive analytics
docker-compose run --rm spotify-pipeline python3 gold/descriptive/build_descriptive_analytics.py

# 4. Gold: Diagnostic analytics
docker-compose run --rm spotify-pipeline python3 gold/diagnostic/build_diagnostic_analytics.py

# 5. Gold: Prescriptive analytics
docker-compose run --rm spotify-pipeline python3 gold/prescriptive/build_prescriptive_analytics.py

# 6. Gold: Cognitive analytics
docker-compose run --rm spotify-pipeline python3 gold/cognitive/build_cognitive_analytics.py

# 7. Gold: Predictive analytics + ML models
docker-compose run --rm spotify-pipeline bash -c \
  "pip3 install --no-cache-dir -q numpy scikit-learn pandas && \
   python3 gold/predictive/build_predictive_models.py"

# 8. Verify in Trino
docker exec -it trino trino --catalog delta --schema default --execute "SHOW TABLES"

# 9. (Optional) Sync to PostgreSQL for Superset
docker-compose run --rm spotify-pipeline python3 scripts/sync_gold_to_postgres.py
```

**Estimated total time:** ~35-40 minutes

---

**Last Updated:** 2025-10-24
**Author:** Claude Code
**Version:** 2.1.0 (Manual Execution Guide)
