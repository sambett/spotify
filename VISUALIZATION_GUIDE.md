# Spotify Analytics Visualization Guide

## 🎯 Quick Access

**Superset Dashboard**: http://localhost:8088
- **Username**: `admin`
- **Password**: `admin`

**Trino Query Interface**: http://localhost:8080

---

## 📊 Available Data & ML Models

### Bronze Layer (Raw Data)
- `listening_history_bronze` - Your recently played tracks from Spotify API
- `my_tracks_features_bronze` - Track metadata (artists, albums, popularity)
- `my_tracks_features_bronze_synthetic` - Synthetic audio features (danceability, energy, valence, tempo)
- `kaggle_tracks_bronze` - 114,000 enrichment tracks from Kaggle

### Silver Layer (Enriched Data)
- `listening_with_features` - Your listening history enriched with audio features
  - Combines real API data + synthetic features + Kaggle data
  - Ready for analysis and visualization

### Gold Layer - Analytics & ML

#### 📈 Descriptive Analytics (What happened?)
- `audio_feature_distributions` - Distribution of danceability, energy, valence across your music
- `feature_source_coverage` - Breakdown of real vs synthetic feature coverage
- `listening_patterns_by_time` - When you listen to music (hourly, daily, weekly)
- `temporal_trends` - How your listening habits evolve over time
- `top_tracks_by_mood` - Most played tracks categorized by mood

#### 🔍 Diagnostic Analytics (Why did it happen?)
- `feature_correlations` - Relationships between audio features
- `mood_shift_patterns` - How your mood changes throughout the day
- `mood_time_correlations` - Correlation between time and mood preferences
- `weekend_vs_weekday` - Different listening patterns on weekends vs weekdays

#### 🤖 Predictive Analytics (What will happen?) - **ML MODELS**
- `mood_predictions` - RandomForest model predicting mood (valence) based on time
  - **Performance**: R² = 0.134, RMSE = 0.173
- `energy_forecasts` - LinearRegression model forecasting energy levels
  - **Performance**: R² = 0.197, RMSE = 0.248
- `mood_classifications` - RandomForest classifier categorizing moods
  - **Performance**: Accuracy = 57%, F1 = 0.51
- `model_performance_metrics` - Detailed metrics for all ML models

#### 💡 Prescriptive Analytics (What should you do?)
- `mood_improvement_recommendations` - Personalized track recommendations to boost mood
  - Example: "Those Eyes" by New West (Valence: 0.78) recommended when feeling down
- `optimal_listening_times` - Best times to listen to certain music types

#### 🧠 Cognitive Analytics (Complex patterns)
- `behavioral_segments` - User behavior clustering (e.g., "Balanced Explorer")
- `listening_anomalies` - Unusual listening patterns detected
- `mood_state_clusters` - K-means clustering of mood states

---

## 🔌 Connecting Superset to Trino

### Step 1: Access Superset
1. Open http://localhost:8088
2. Login with `admin` / `admin`

### Step 2: Add Trino Database Connection
1. Click **Settings** → **Database Connections**
2. Click **+ Database**
3. Select **Trino** from the list
4. Use this connection string:

```
trino://admin@trino:8080/delta/default
```

5. Click **Test Connection** (should show ✅)
6. Click **Connect**

### Step 3: Register Tables
1. Go to **Data** → **Datasets**
2. Click **+ Dataset**
3. Select your Trino database
4. Choose tables to add (start with these key tables):
   - `listening_with_features`
   - `mood_predictions`
   - `mood_improvement_recommendations`
   - `behavioral_segments`
   - `listening_patterns_by_time`

---

## 📊 Recommended Visualizations

### 1. **Mood Over Time Dashboard**
**Goal**: Visualize how your mood changes throughout the day/week

**Charts to create**:
- Line chart: `mood_predictions` - predicted_valence vs hour_of_day
- Heatmap: `mood_time_correlations` - mood intensity by hour and day
- Bar chart: `weekend_vs_weekday` - average mood comparison

**SQL Example**:
```sql
SELECT
    hour_of_day,
    AVG(predicted_valence) as avg_mood,
    AVG(energy) as avg_energy
FROM mood_predictions
GROUP BY hour_of_day
ORDER BY hour_of_day
```

### 2. **Music Feature Analysis**
**Goal**: Understand audio characteristics of your music

**Charts to create**:
- Histogram: `audio_feature_distributions` - danceability distribution
- Scatter plot: `feature_correlations` - energy vs valence
- Pie chart: `feature_source_coverage` - real vs synthetic features

**SQL Example**:
```sql
SELECT
    valence,
    energy,
    danceability,
    tempo
FROM listening_with_features
LIMIT 1000
```

### 3. **ML Model Performance**
**Goal**: Show accuracy and performance of predictive models

**Charts to create**:
- Table: `model_performance_metrics` - all metrics
- Bar chart: R² scores by model
- Scatter: actual vs predicted values from `mood_predictions`

**SQL Example**:
```sql
SELECT
    model_name,
    model_type,
    r2_score,
    accuracy,
    f1_score
FROM model_performance_metrics
```

### 4. **Personalized Recommendations**
**Goal**: Display ML-generated music recommendations

**Charts to create**:
- Table: `mood_improvement_recommendations` - top recommendations
- Bar chart: avg_valence by recommendation reason
- Word cloud: target_scenario distribution

**SQL Example**:
```sql
SELECT
    track_name,
    artist_name,
    avg_valence,
    avg_energy,
    recommendation_reason,
    target_scenario
FROM mood_improvement_recommendations
ORDER BY avg_valence DESC
LIMIT 10
```

### 5. **Behavioral Insights**
**Goal**: Understand your listening personality

**Charts to create**:
- Gauge: `behavioral_segments` - your segment characteristics
- Radar chart: avg_valence, avg_energy, avg_danceability for your segment
- Sankey: listening flow throughout the day

---

## 🎨 Example Queries for Insights

### Your Most Energetic Hours
```sql
SELECT
    hour_of_day,
    AVG(energy) as avg_energy,
    COUNT(*) as play_count
FROM listening_with_features
GROUP BY hour_of_day
ORDER BY avg_energy DESC
```

### Mood Boosting Playlist
```sql
SELECT DISTINCT
    track_name,
    artist_name,
    avg_valence,
    recommendation_reason
FROM mood_improvement_recommendations
WHERE target_scenario = 'When feeling down or low energy'
ORDER BY avg_valence DESC
LIMIT 20
```

### Listening Anomalies
```sql
SELECT *
FROM listening_anomalies
ORDER BY anomaly_score DESC
LIMIT 10
```

### Your Music Personality
```sql
SELECT
    segment_name,
    interpretation,
    recommendation,
    avg_valence,
    avg_energy,
    avg_danceability
FROM behavioral_segments
```

---

## 🚀 Next Steps

1. **Access Superset**: http://localhost:8088 (admin/admin)
2. **Connect to Trino**: Use connection string above
3. **Create your first chart**: Start with "Mood Over Time"
4. **Build a dashboard**: Combine multiple charts
5. **Explore ML insights**: Check model performance and predictions
6. **Get recommendations**: Use the mood_improvement_recommendations table

---

## 📅 Data Updates

Your data refreshes automatically:
- **Schedule**: Every 6 hours (4x daily)
- **Next run**: Check `docker logs spotify-scheduler`
- **What updates**:
  - New listening history from Spotify API
  - Fresh synthetic features
  - Retrained ML models
  - Updated analytics

---

## 🔧 Troubleshooting

### Can't connect to Trino in Superset?
```bash
# Check if Trino is running
docker ps | grep trino

# Test Trino manually
docker exec trino trino --execute "SHOW TABLES FROM delta.default"
```

### Need to see raw data?
```bash
# Query directly via Trino CLI
docker exec -it trino trino

# Then run:
USE delta.default;
SHOW TABLES;
SELECT * FROM mood_predictions LIMIT 5;
```

### Want to trigger pipeline manually?
```bash
# Run pipeline now (instead of waiting 6 hours)
docker exec spotify-scheduler python3 run_full_pipeline.py
```

---

## 📧 Support

For issues or questions, check:
- Scheduler logs: `docker logs spotify-scheduler`
- Superset logs: `docker logs superset`
- Trino logs: `docker logs trino`
