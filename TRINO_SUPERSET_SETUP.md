# Trino + Superset Setup Guide

## Why Use Trino Instead of PostgreSQL?

| Feature | Trino | PostgreSQL |
|---------|-------|------------|
| **Data Location** | Queries Delta Lake directly | Requires copying data |
| **Storage** | No duplication | Doubles storage |
| **Freshness** | Always queries source of truth | Stale until re-synced |
| **Performance** | Optimized for analytics | Optimized for transactions |
| **Scalability** | Distributed queries | Single node |

**Recommendation:** Use Trino for all analytics queries in Superset.

---

## Step-by-Step Setup

### 1. Access Superset

Open your browser and go to: **http://localhost:8088**

**Login credentials:**
- Username: `admin`
- Password: `admin`

### 2. Add Trino Database Connection

1. Click **Settings** (gear icon in top right)
2. Select **Database Connections**
3. Click **+ Database** button
4. Choose **Trino** from the list

### 3. Configure Trino Connection

**SUPPORTED DATABASE:**
- Select: **Trino**

**SQLALCHEMY URI:**
```
trino://trino:8080/delta/default
```

**Display Name:**
```
Spotify Analytics (Trino)
```

**ADVANCED** (Optional):
- **Expose database in SQL Lab**: ✅ Check this
- **Allow CREATE TABLE AS**: ✅ Check this (for derived datasets)
- **Allow DML**: ❌ Leave unchecked (read-only for safety)

### 4. Test Connection

Click **Test Connection** button.

You should see: ✅ **Connection looks good!**

If you see an error, verify:
- Trino container is running: `docker ps | grep trino`
- Trino is healthy: `docker logs trino --tail 50`

### 5. Save Connection

Click **Connect** to save the database connection.

---

## Available Tables in Trino

Once connected, you can query these tables:

### Bronze Layer (Raw Data)
```sql
SELECT * FROM delta.default.listening_history_bronze LIMIT 10;
SELECT * FROM delta.default.kaggle_tracks_bronze LIMIT 10;
SELECT * FROM delta.default.my_tracks_features_bronze_synthetic LIMIT 10;
```

### Silver Layer (Enriched Data)
```sql
SELECT * FROM delta.default.listening_with_features LIMIT 10;
```

### Gold Layer - Descriptive Analytics
```sql
SELECT * FROM delta.default.listening_patterns_by_time;
SELECT * FROM delta.default.top_tracks_by_mood;
SELECT * FROM delta.default.temporal_trends;
SELECT * FROM delta.default.audio_feature_distributions;
SELECT * FROM delta.default.feature_source_coverage;
```

### Gold Layer - Diagnostic Analytics
```sql
SELECT * FROM delta.default.mood_time_correlations;
SELECT * FROM delta.default.feature_correlations;
SELECT * FROM delta.default.weekend_vs_weekday;
SELECT * FROM delta.default.mood_shift_patterns;
SELECT * FROM delta.default.part_of_day_drivers;
```

### Gold Layer - Predictive Analytics
```sql
SELECT * FROM delta.default.mood_predictions;
SELECT * FROM delta.default.energy_forecasts;
SELECT * FROM delta.default.mood_classifications;
SELECT * FROM delta.default.model_performance_metrics;
```

### Gold Layer - Prescriptive Analytics
```sql
SELECT * FROM delta.default.mood_improvement_recommendations;
SELECT * FROM delta.default.optimal_listening_times;
SELECT * FROM delta.default.personalized_playlist_suggestions;
SELECT * FROM delta.default.mood_intervention_triggers;
```

### Gold Layer - Cognitive Analytics
```sql
SELECT * FROM delta.default.mood_state_clusters;
SELECT * FROM delta.default.listening_anomalies;
SELECT * FROM delta.default.sequential_patterns;
SELECT * FROM delta.default.behavioral_segments;
```

---

## How to Create Datasets

### Method 1: Auto-Sync Tables

1. Go to **Data** → **Datasets**
2. Click **+ Dataset** button
3. Select:
   - **Database**: Spotify Analytics (Trino)
   - **Schema**: default
   - **Table**: Choose any table from the dropdown
4. Click **Create Dataset and Create Chart**

### Method 2: SQL Query Dataset (Recommended for Complex Queries)

1. Go to **SQL Lab** → **SQL Editor**
2. Select database: **Spotify Analytics (Trino)**
3. Write your query (see examples below)
4. Click **Run**
5. Once results appear, click **Save** → **Save Dataset**

---

## Example Queries for Charts

### 1. Listening Patterns by Hour (Fixed Chart)

**Problem:** The chart you showed summed pre-aggregated data.

**Solution:** Query the Silver layer directly for individual events:

```sql
SELECT
    hour_of_day,
    COUNT(*) as total_plays,
    AVG(valence) as avg_mood,
    AVG(energy) as avg_energy
FROM delta.default.listening_with_features
GROUP BY hour_of_day
ORDER BY hour_of_day
```

**Chart Type:** Line Chart
- **X-Axis:** hour_of_day
- **Metrics:** total_plays, avg_mood, avg_energy

---

### 2. Mood Distribution Over Time

```sql
SELECT
    hour_of_day,
    CASE
        WHEN valence >= 0.6 AND energy >= 0.6 THEN 'Happy & Energetic'
        WHEN valence >= 0.6 AND energy < 0.4 THEN 'Happy & Calm'
        WHEN valence < 0.4 AND energy >= 0.6 THEN 'Sad & Energetic'
        WHEN valence < 0.4 AND energy < 0.4 THEN 'Sad & Calm'
        ELSE 'Neutral'
    END as mood_category,
    COUNT(*) as count
FROM delta.default.listening_with_features
GROUP BY hour_of_day, mood_category
ORDER BY hour_of_day, mood_category
```

**Chart Type:** Stacked Bar Chart
- **X-Axis:** hour_of_day
- **Metrics:** count
- **Group By:** mood_category

---

### 3. Top Artists by Play Count

```sql
SELECT
    artist_name,
    COUNT(*) as play_count,
    AVG(valence) as avg_mood,
    AVG(energy) as avg_energy
FROM delta.default.listening_with_features
GROUP BY artist_name
ORDER BY play_count DESC
LIMIT 20
```

**Chart Type:** Bar Chart
- **X-Axis:** artist_name
- **Metrics:** play_count

---

### 4. Weekend vs Weekday Listening

```sql
SELECT
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END as day_type,
    COUNT(*) as total_plays,
    AVG(valence) as avg_mood,
    AVG(energy) as avg_energy,
    AVG(danceability) as avg_danceability
FROM delta.default.listening_with_features
GROUP BY is_weekend
```

**Chart Type:** Big Number with Trendline
- Compare weekend vs weekday metrics

---

### 5. Model Performance Dashboard

```sql
SELECT
    model_name,
    model_type,
    training_approach,
    COALESCE(r2, 0) as r2_score,
    COALESCE(rmse, 0) as rmse_score,
    COALESCE(accuracy, 0) as accuracy_score,
    train_size,
    test_size
FROM delta.default.model_performance_metrics
```

**Chart Type:** Table
- Shows all ML model metrics side-by-side

---

### 6. Mood Prediction Accuracy

```sql
SELECT
    hour_of_day,
    AVG(valence) as actual_mood,
    AVG(predicted_valence) as predicted_mood,
    AVG(ABS(valence - predicted_valence)) as avg_error
FROM delta.default.mood_predictions
GROUP BY hour_of_day
ORDER BY hour_of_day
```

**Chart Type:** Line Chart (Dual Axis)
- **X-Axis:** hour_of_day
- **Y-Axis (Left):** actual_mood, predicted_mood
- **Y-Axis (Right):** avg_error

---

## Troubleshooting

### Trino connection fails

**Check Trino status:**
```bash
docker ps | grep trino
docker logs trino --tail 50
```

**Restart Trino:**
```bash
docker-compose restart trino
```

### Tables not showing up

**Verify Delta tables exist:**
```bash
docker exec -it trino trino --catalog delta --schema default
```

Then run:
```sql
SHOW TABLES;
```

### Query timeout

Increase timeout in Superset:
1. Go to **Settings** → **Database Connections**
2. Edit Trino connection
3. **Advanced** → **SQL Lab** section
4. Increase **Query timeout** to 300 seconds

---

## Best Practices

1. **Use Trino for Analytics**
   - Query Silver/Gold layers directly
   - No need to sync to PostgreSQL

2. **PostgreSQL is for Superset Metadata Only**
   - Keep it for Superset's own data
   - Don't use `sync_gold_to_postgres.py` unless needed for specific use case

3. **Cache Query Results**
   - Superset uses Redis to cache query results
   - Charts load faster on repeat views

4. **Use SQL Lab for Complex Queries**
   - Test queries in SQL Lab first
   - Save as dataset once validated
   - Create charts from dataset

5. **Create Filtered Datasets**
   - Filter data at query time for specific use cases
   - Example: "Last 30 days listening history"

---

## Next Steps

1. ✅ Connect Superset to Trino
2. 🎨 Create datasets from Gold layer tables
3. 📊 Build visualizations for each analytics type:
   - Descriptive: What happened?
   - Diagnostic: Why did it happen?
   - Predictive: What will happen?
   - Prescriptive: What should we do?
   - Cognitive: Complex patterns
4. 📈 Combine charts into comprehensive dashboards
5. 🔄 Set up automatic refresh schedules for charts

---

## Useful Commands

### Test Trino CLI
```bash
docker exec -it trino trino --catalog delta --schema default
```

### Query tables from CLI
```sql
SHOW TABLES;
SELECT * FROM listening_with_features LIMIT 5;
DESCRIBE listening_patterns_by_time;
```

### Check Trino connector config
```bash
docker exec -it trino cat /etc/trino/catalog/delta.properties
```

---

**Questions or issues?** Check the Trino logs and Superset logs for errors.
