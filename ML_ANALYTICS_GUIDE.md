# 🤖 Machine Learning & Analytics Complete Guide

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR WORKFLOW                             │
└─────────────────────────────────────────────────────────────┘

1. DATA COLLECTION (Automated Every 6 Hours)
   ├── Spotify API → Recently Played Songs
   ├── Kaggle Dataset → 114,000 enrichment tracks
   └── Synthetic Generator → Audio features for your tracks

2. DATA STORAGE (Delta Lake)
   ├── Bronze Layer: Raw data
   ├── Silver Layer: Cleaned & enriched
   └── Gold Layer: ML predictions & analytics

3. QUERY ENGINE (Trino)
   └── Reads Delta Lake tables directly (no copying!)

4. VISUALIZATION (Superset)
   └── Connects to Trino to create charts & dashboards
```

---

## 🔍 How Trino Works

### What is Trino?
Trino is a **distributed SQL query engine** that can query data where it lives - no need to move data into a database!

### Why We Use Trino:
✅ **Direct Delta Lake Access**: Queries parquet files directly from `/data`
✅ **No Data Duplication**: Your data stays in Delta Lake format
✅ **Fast Queries**: Optimized for analytics workloads
✅ **Standard SQL**: Use familiar SQL syntax
✅ **Multiple Data Sources**: Can join Delta Lake + Postgres + more

### How It Works:
```
Superset Query:
"SELECT * FROM mood_predictions WHERE hour_of_day = 14"
              ↓
         Trino Engine
              ↓
    Reads from /data/gold/predictive/mood_predictions/*.parquet
              ↓
    Returns results to Superset
              ↓
    Superset visualizes the data
```

### Connection Details:
- **URL**: `trino://admin@trino:8080/delta/default`
- **Catalog**: `delta` (Delta Lake connector)
- **Schema**: `default` (your data)
- **Tables**: 22 tables available for querying

---

## 🤖 Machine Learning Models Explained

### Model 1: Mood Predictor (Valence Prediction)
**Type**: Random Forest Regressor
**Purpose**: Predict your mood based on time of day
**Performance**: R² = 0.134 (13.4% variance explained)

**How It Works**:
```python
Input:
  - hour_of_day (0-23)
  - day_of_week (0-6)
  - Historical listening patterns

Output:
  - predicted_valence (0.0-1.0)
    0.0 = Sad/Low mood
    0.5 = Neutral
    1.0 = Happy/High mood

Training Data:
  - 1,504 your listening records
  - 113,692 Kaggle reference tracks
```

**Table**: `mood_predictions`
**Location**: `/data/gold/predictive/mood_predictions/`

**Example Query**:
```sql
SELECT
    hour_of_day,
    AVG(predicted_valence) as avg_mood,
    AVG(energy) as avg_energy
FROM mood_predictions
GROUP BY hour_of_day
ORDER BY hour_of_day
```

---

### Model 2: Energy Forecaster
**Type**: Linear Regression
**Purpose**: Predict energy levels in your music choices
**Performance**: R² = 0.197 (19.7% variance explained)

**How It Works**:
```python
Input:
  - Time features
  - Historical energy patterns
  - Listening frequency

Output:
  - predicted_energy (0.0-1.0)
    0.0 = Low energy (calm, relaxed)
    1.0 = High energy (intense, upbeat)
```

**Table**: `energy_forecasts`
**Location**: `/data/gold/predictive/energy_forecasts/`

**Example Query**:
```sql
SELECT
    hour_of_day,
    predicted_energy,
    actual_energy,
    ABS(predicted_energy - actual_energy) as prediction_error
FROM energy_forecasts
ORDER BY prediction_error ASC
LIMIT 10
```

---

### Model 3: Mood Classifier
**Type**: Random Forest Classifier
**Purpose**: Categorize your music into mood categories
**Performance**: Accuracy = 57%, F1 = 0.51

**How It Works**:
```python
Input:
  - valence (happiness)
  - energy (intensity)
  - danceability
  - tempo

Output:
  - mood_category:
    - "Happy & Energetic"
    - "Calm & Peaceful"
    - "Melancholic"
    - "Intense & Dark"
    - etc.
```

**Table**: `mood_classifications`
**Location**: `/data/gold/predictive/mood_classifications/`

---

### Model Performance Metrics
**Table**: `model_performance_metrics`

**Query to See All Models**:
```sql
SELECT
    model_name,
    model_type,
    r2_score,
    rmse,
    accuracy,
    f1_score,
    train_size,
    test_size
FROM model_performance_metrics
```

**Results**:
| Model | Type | R² Score | Accuracy | F1 Score |
|-------|------|----------|----------|----------|
| mood_prediction | RandomForest | 0.134 | - | - |
| energy_forecast | LinearRegression | 0.197 | - | - |
| mood_classifier | RandomForest | - | 0.570 | 0.509 |

---

## 📊 All Available Analytics Tables

### 📈 Descriptive Analytics (What Happened?)

#### 1. `listening_with_features` (MOST IMPORTANT)
Your complete listening history with all audio features merged.

**Columns**:
- `track_id`, `track_name`, `artist_name`, `album_name`
- `played_at` (timestamp)
- `danceability`, `energy`, `valence`, `tempo`
- `acousticness`, `instrumentalness`, `speechiness`
- `feature_source` (real/synthetic/kaggle)

**Example Query**:
```sql
SELECT
    track_name,
    artist_name,
    played_at,
    valence as happiness,
    energy,
    danceability
FROM listening_with_features
ORDER BY played_at DESC
LIMIT 20
```

#### 2. `audio_feature_distributions`
Statistical distribution of audio features in your music.

**Example Query**:
```sql
SELECT * FROM audio_feature_distributions
```

#### 3. `listening_patterns_by_time`
When you listen to music (hourly, daily, weekly patterns).

**Example Query**:
```sql
SELECT
    hour_of_day,
    day_of_week,
    play_count,
    avg_valence
FROM listening_patterns_by_time
ORDER BY play_count DESC
```

#### 4. `top_tracks_by_mood`
Your most played songs categorized by mood.

---

### 🔍 Diagnostic Analytics (Why It Happened?)

#### 5. `feature_correlations`
How audio features relate to each other.

**Example Query**:
```sql
SELECT
    feature_pair,
    correlation_value
FROM feature_correlations
WHERE ABS(correlation_value) > 0.3
ORDER BY ABS(correlation_value) DESC
```

#### 6. `mood_shift_patterns`
How your mood changes throughout the day/week.

**Example Query**:
```sql
SELECT
    from_mood,
    to_mood,
    transition_count,
    avg_time_between_hours
FROM mood_shift_patterns
ORDER BY transition_count DESC
```

#### 7. `weekend_vs_weekday`
Differences in listening between weekdays and weekends.

---

### 💡 Prescriptive Analytics (What Should You Do?)

#### 8. `mood_improvement_recommendations` ⭐
**Personalized track recommendations to boost your mood!**

**Columns**:
- `track_name`, `artist_name`
- `avg_valence`, `avg_energy`, `avg_danceability`
- `play_count`
- `recommendation_reason` ("Positive track to enhance wellbeing")
- `target_scenario` ("When feeling down or low energy")

**Example Query**:
```sql
SELECT
    track_name,
    artist_name,
    avg_valence as happiness_score,
    recommendation_reason
FROM mood_improvement_recommendations
WHERE target_scenario = 'When feeling down or low energy'
ORDER BY avg_valence DESC
LIMIT 10
```

**Sample Results**:
| Track | Artist | Happiness | Reason |
|-------|--------|-----------|--------|
| Those Eyes | New West | 0.784 | Positive track to enhance wellbeing |
| Wild Time | Weyes Blood | 0.781 | High energy track to boost mood |

#### 9. `optimal_listening_times`
Best times to listen to certain music types.

---

### 🧠 Cognitive Analytics (Complex Patterns)

#### 10. `behavioral_segments` ⭐
**Your listening personality classification!**

**Columns**:
- `avg_valence`, `avg_energy`, `avg_danceability`
- `segment_name` ("Balanced Explorer", "Energy Seeker", etc.)
- `interpretation` (personality description)
- `recommendation` (advice for your listening style)

**Example Query**:
```sql
SELECT
    segment_name,
    interpretation,
    recommendation,
    avg_valence,
    avg_energy
FROM behavioral_segments
```

**Sample Result**:
```
Segment: Balanced Explorer
Interpretation: Varied musical taste. Adapts music to different moods and contexts.
Recommendation: Healthy listening patterns - continue current approach
```

#### 11. `listening_anomalies`
Unusual listening patterns detected by the system.

#### 12. `mood_state_clusters`
K-means clustering of your different mood states.

---

## 🎨 Creating Visualizations in Superset

### Step 1: Add Datasets

1. Go to **Data** → **Datasets**
2. Click **+ Dataset**
3. Database: "Spotify Analytics" (your Trino connection)
4. Schema: `default`
5. Table: Select a table (start with these):
   - `listening_with_features`
   - `mood_predictions`
   - `mood_improvement_recommendations`
   - `behavioral_segments`

### Step 2: Create Your First Chart

#### Chart 1: Mood Throughout the Day 📈

1. Go to **Charts** → **+ Chart**
2. Choose dataset: `mood_predictions`
3. Chart type: **Line Chart**
4. Configuration:
   - **Dimensions** (X-Axis): `hour_of_day`
   - **Metrics** (Y-Axis): `AVG(predicted_valence)`
   - **Chart Title**: "My Mood Pattern Throughout the Day"
5. Click **Update Chart**
6. Save: "Mood Over Time"

**What You'll See**: A line showing how your predicted mood changes from 0am to 11pm!

---

#### Chart 2: Energy vs Happiness Scatter 🎯

1. Create new chart
2. Dataset: `listening_with_features`
3. Chart type: **Scatter Plot**
4. Configuration:
   - **X-Axis**: `energy`
   - **Y-Axis**: `valence`
   - **Size**: `COUNT(*)`
5. **What You'll See**: Clustering of your music taste!

---

#### Chart 3: Top Mood-Boosting Tracks 🎵

1. Create new chart
2. Dataset: `mood_improvement_recommendations`
3. Chart type: **Table**
4. Columns to show:
   - `track_name`
   - `artist_name`
   - `avg_valence` (rename to "Happiness Score")
   - `recommendation_reason`
5. Sort by: `avg_valence` DESC
6. **What You'll See**: Your personalized mood-boosting playlist!

---

#### Chart 4: Listening Personality 🧠

1. Create new chart
2. Dataset: `behavioral_segments`
3. Chart type: **Radar Chart** or **Big Number**
4. Show:
   - `segment_name`
   - `interpretation`
   - Radar: `avg_valence`, `avg_energy`, `avg_danceability`
5. **What You'll See**: Your music personality profile!

---

#### Chart 5: ML Model Performance 📊

1. Dataset: `model_performance_metrics`
2. Chart type: **Table**
3. Show all columns
4. **What You'll See**: How accurate your ML models are!

---

### Step 3: Create a Dashboard

1. Go to **Dashboards** → **+ Dashboard**
2. Name: "My Spotify Analytics"
3. Drag and drop charts:
   ```
   ┌─────────────────┬─────────────────┐
   │ Mood Over Time  │ Energy Scatter  │
   ├─────────────────┴─────────────────┤
   │   Mood Boosting Recommendations   │
   ├───────────────────────────────────┤
   │        Listening Personality      │
   └───────────────────────────────────┘
   ```
4. Save dashboard

---

## 📝 Useful SQL Queries

### Your Listening Summary
```sql
SELECT
    COUNT(DISTINCT track_id) as unique_tracks,
    COUNT(*) as total_plays,
    AVG(valence) as avg_happiness,
    AVG(energy) as avg_energy,
    AVG(danceability) as avg_danceability
FROM listening_with_features
```

### Most Played Artists
```sql
SELECT
    artist_name,
    COUNT(*) as play_count,
    AVG(valence) as avg_mood
FROM listening_with_features
GROUP BY artist_name
ORDER BY play_count DESC
LIMIT 10
```

### Mood by Hour
```sql
SELECT
    EXTRACT(HOUR FROM played_at) as hour,
    AVG(valence) as avg_mood,
    COUNT(*) as plays
FROM listening_with_features
GROUP BY EXTRACT(HOUR FROM played_at)
ORDER BY hour
```

### Weekend Party Tracks
```sql
SELECT
    track_name,
    artist_name,
    energy,
    danceability
FROM listening_with_features
WHERE energy > 0.7
  AND danceability > 0.7
GROUP BY track_name, artist_name, energy, danceability
ORDER BY energy DESC
LIMIT 20
```

---

## 🔄 Data Flow Summary

```
Every 6 Hours (Automated):
├── 1. Fetch recently played from Spotify API
├── 2. Generate synthetic features for new tracks
├── 3. Save to Bronze layer (Delta Lake)
├── 4. Enrich with Kaggle data → Silver layer
├── 5. Run ML models:
│   ├── Train mood predictor
│   ├── Train energy forecaster
│   └── Train mood classifier
├── 6. Generate recommendations
├── 7. Cluster listening behavior
└── 8. Save all to Gold layer

Then You Can:
├── Query via Trino (SQL)
├── Visualize in Superset
└── Build custom dashboards
```

---

## 🎯 Quick Access

- **Superset**: http://localhost:8088 (admin/admin)
- **Trino Web UI**: http://localhost:8080
- **Connection**: `trino://admin@trino:8080/delta/default`

## 📅 Next Pipeline Run

Check when next data refresh happens:
```bash
docker logs spotify-scheduler | grep "Schedule"
```

Your data auto-updates every 6 hours! 🎉
