# Superset Query Library

Ready-to-use SQL queries for creating charts in Apache Superset.

**Database:** Spotify Analytics (Trino)
**Connection:** `trino://trino:8080/delta/default`

---

## 📊 Descriptive Analytics (What Happened?)

### Q1: Hourly Listening Activity

**Purpose:** See when you listen to music most
**Chart Type:** Line Chart

```sql
SELECT
    hour_of_day,
    COUNT(*) as total_plays,
    AVG(valence) as avg_happiness,
    AVG(energy) as avg_energy,
    AVG(danceability) as avg_danceability
FROM delta.default.listening_with_features
GROUP BY hour_of_day
ORDER BY hour_of_day
```

**Chart Config:**
- X-Axis: hour_of_day
- Metrics: total_plays (primary), avg_happiness, avg_energy (secondary axis)

---

### Q2: Top 20 Artists

**Purpose:** Most played artists
**Chart Type:** Horizontal Bar Chart

```sql
SELECT
    artist_name,
    COUNT(*) as play_count,
    AVG(valence) as avg_mood
FROM delta.default.listening_with_features
GROUP BY artist_name
ORDER BY play_count DESC
LIMIT 20
```

**Chart Config:**
- X-Axis: play_count
- Y-Axis: artist_name
- Sort: Descending by play_count

---

### Q3: Top Tracks by Mood Category

**Purpose:** Most played songs in each mood
**Chart Type:** Table

```sql
WITH mood_categories AS (
    SELECT
        track_name,
        artist_name,
        CASE
            WHEN valence >= 0.6 AND energy >= 0.6 THEN 'Happy & Energetic'
            WHEN valence >= 0.6 AND energy < 0.4 THEN 'Happy & Calm'
            WHEN valence < 0.4 AND energy >= 0.6 THEN 'Sad & Energetic'
            WHEN valence < 0.4 AND energy < 0.4 THEN 'Sad & Calm'
            ELSE 'Neutral'
        END as mood,
        COUNT(*) as plays
    FROM delta.default.listening_with_features
    GROUP BY track_name, artist_name, mood
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY mood ORDER BY plays DESC) as rank
    FROM mood_categories
)
SELECT mood, track_name, artist_name, plays
FROM ranked
WHERE rank <= 5
ORDER BY mood, plays DESC
```

---

### Q4: Listening Patterns by Day of Week

**Purpose:** Compare listening habits across days
**Chart Type:** Heatmap

```sql
SELECT
    day_of_week,
    hour_of_day,
    COUNT(*) as play_count
FROM delta.default.listening_with_features
GROUP BY day_of_week, hour_of_day
ORDER BY day_of_week, hour_of_day
```

**Chart Config:**
- X-Axis: hour_of_day (0-23)
- Y-Axis: day_of_week (0=Sunday, 6=Saturday)
- Metric: play_count (color intensity)

---

### Q5: Audio Feature Distribution

**Purpose:** Understand music preferences
**Chart Type:** Box Plot or Histogram

```sql
SELECT
    'Valence (Happiness)' as feature,
    valence as value
FROM delta.default.listening_with_features
WHERE valence IS NOT NULL

UNION ALL

SELECT
    'Energy',
    energy
FROM delta.default.listening_with_features
WHERE energy IS NOT NULL

UNION ALL

SELECT
    'Danceability',
    danceability
FROM delta.default.listening_with_features
WHERE danceability IS NOT NULL

UNION ALL

SELECT
    'Acousticness',
    acousticness
FROM delta.default.listening_with_features
WHERE acousticness IS NOT NULL
```

---

## 🔍 Diagnostic Analytics (Why Did It Happen?)

### Q6: Weekend vs Weekday Comparison

**Purpose:** How does listening differ on weekends?
**Chart Type:** Big Number with Trendline (2 charts side by side)

```sql
SELECT
    CASE
        WHEN is_weekend THEN 'Weekend'
        ELSE 'Weekday'
    END as day_type,
    COUNT(*) as total_plays,
    AVG(valence) as avg_happiness,
    AVG(energy) as avg_energy,
    AVG(tempo) as avg_tempo
FROM delta.default.listening_with_features
GROUP BY is_weekend
```

---

### Q7: Mood Shift Throughout Day

**Purpose:** Track how mood changes by time of day
**Chart Type:** Area Chart

```sql
SELECT
    hour_of_day,
    AVG(valence) as avg_valence,
    AVG(energy) as avg_energy,
    COUNT(*) as sample_size
FROM delta.default.listening_with_features
GROUP BY hour_of_day
ORDER BY hour_of_day
```

**Chart Config:**
- X-Axis: hour_of_day
- Metrics: avg_valence (stacked), avg_energy (stacked)
- Show sample_size as tooltip

---

### Q8: Feature Correlation Matrix

**Purpose:** Which audio features correlate?
**Chart Type:** Table with Conditional Formatting

```sql
SELECT
    feature1,
    feature2,
    correlation
FROM delta.default.feature_correlations
ORDER BY ABS(correlation) DESC
```

---

### Q9: Part of Day Analysis

**Purpose:** Morning vs Afternoon vs Evening vs Night patterns
**Chart Type:** Grouped Bar Chart

```sql
SELECT
    part_of_day,
    AVG(valence) as avg_happiness,
    AVG(energy) as avg_energy,
    AVG(danceability) as avg_danceability,
    AVG(acousticness) as avg_acousticness,
    COUNT(*) as play_count
FROM delta.default.listening_with_features
WHERE part_of_day IS NOT NULL
GROUP BY part_of_day
ORDER BY
    CASE part_of_day
        WHEN 'morning' THEN 1
        WHEN 'afternoon' THEN 2
        WHEN 'evening' THEN 3
        WHEN 'night' THEN 4
    END
```

---

### Q10: Mood-Time Correlation

**Purpose:** When are you happiest?
**Chart Type:** Scatter Plot

```sql
SELECT
    hour_of_day,
    valence,
    energy,
    track_name,
    artist_name
FROM delta.default.listening_with_features
WHERE valence IS NOT NULL AND energy IS NOT NULL
```

**Chart Config:**
- X-Axis: hour_of_day
- Y-Axis: valence
- Size: energy
- Tooltip: track_name, artist_name

---

## 🔮 Predictive Analytics (What Will Happen?)

### Q11: Model Performance Comparison

**Purpose:** How accurate are the ML models?
**Chart Type:** Table

```sql
SELECT
    model_name,
    model_type,
    training_approach,
    CAST(train_size AS VARCHAR) || ' tracks' as training_data,
    CAST(test_size AS VARCHAR) || ' tracks' as test_data,
    ROUND(COALESCE(r2, 0), 4) as r2_score,
    ROUND(COALESCE(rmse, 0), 4) as rmse,
    ROUND(COALESCE(mae, 0), 4) as mae,
    ROUND(COALESCE(accuracy, 0) * 100, 2) || '%' as accuracy,
    ROUND(COALESCE(f1, 0), 4) as f1_score
FROM delta.default.model_performance_metrics
```

---

### Q12: Actual vs Predicted Mood

**Purpose:** How well does the model predict your mood?
**Chart Type:** Line Chart (Dual Axis)

```sql
SELECT
    hour_of_day,
    AVG(valence) as actual_mood,
    AVG(predicted_valence) as predicted_mood,
    AVG(ABS(valence - predicted_valence)) as prediction_error,
    COUNT(*) as data_points
FROM delta.default.mood_predictions
GROUP BY hour_of_day
ORDER BY hour_of_day
```

**Chart Config:**
- X-Axis: hour_of_day
- Y-Axis (Left): actual_mood, predicted_mood
- Y-Axis (Right): prediction_error
- Annotations: Show where error is high

---

### Q13: Mood Classification Distribution

**Purpose:** Distribution of predicted mood categories
**Chart Type:** Pie Chart

```sql
SELECT
    CASE predicted_mood_label
        WHEN 0 THEN 'Happy & Energetic'
        WHEN 1 THEN 'Happy & Calm'
        WHEN 2 THEN 'Sad & Energetic'
        WHEN 3 THEN 'Sad & Calm'
        ELSE 'Neutral'
    END as predicted_mood,
    COUNT(*) as count
FROM delta.default.mood_classifications
GROUP BY predicted_mood_label
ORDER BY count DESC
```

---

## 💡 Prescriptive Analytics (What Should We Do?)

### Q14: Mood Intervention Triggers

**Purpose:** When do you need a mood boost?
**Chart Type:** Table with Alert Styling

```sql
SELECT
    hour_of_day,
    ROUND(avg_valence, 3) as avg_mood,
    ROUND(avg_energy, 3) as avg_energy,
    low_mood_count,
    total_plays,
    ROUND(CAST(low_mood_count AS DOUBLE) / CAST(total_plays AS DOUBLE) * 100, 1) || '%' as low_mood_percentage,
    intervention_needed,
    suggested_intervention,
    mood_support_playlist
FROM delta.default.mood_intervention_triggers
ORDER BY
    CASE intervention_needed
        WHEN 'High Priority' THEN 1
        WHEN 'Medium Priority' THEN 2
        WHEN 'Low Priority' THEN 3
    END,
    hour_of_day
```

---

### Q15: Mood Improvement Recommendations

**Purpose:** What songs improve your mood?
**Chart Type:** Table

```sql
SELECT
    track_name,
    artist_name,
    play_count,
    ROUND(avg_valence, 3) as happiness,
    ROUND(avg_energy, 3) as energy,
    recommendation_reason,
    target_scenario
FROM delta.default.mood_improvement_recommendations
ORDER BY avg_valence DESC, play_count DESC
```

---

### Q16: Optimal Listening Times

**Purpose:** When should you listen to music?
**Chart Type:** Timeline / Gantt Chart

```sql
SELECT
    hour_of_day,
    mood_state,
    ROUND(predicted_valence, 3) as expected_happiness,
    ROUND(avg_energy, 3) as expected_energy,
    recommended_activity,
    wellbeing_tip
FROM delta.default.optimal_listening_times
ORDER BY hour_of_day
```

---

### Q17: Personalized Playlist Suggestions

**Purpose:** Curated playlists based on your listening
**Chart Type:** Table grouped by playlist

```sql
SELECT
    playlist_name,
    playlist_purpose,
    COUNT(*) as track_count,
    STRING_AGG(track_name || ' - ' || artist_name, ', ') as tracks
FROM delta.default.personalized_playlist_suggestions
GROUP BY playlist_name, playlist_purpose
ORDER BY playlist_name
```

---

## 🧠 Cognitive Analytics (Complex Patterns)

### Q18: Mood State Clusters

**Purpose:** Discover your listening personas
**Chart Type:** Scatter Plot (3D if possible)

```sql
SELECT
    cluster,
    mood_state_name,
    member_count,
    ROUND(avg_valence, 3) as avg_happiness,
    ROUND(avg_energy, 3) as avg_energy,
    ROUND(avg_danceability, 3) as avg_danceability,
    cluster_description
FROM delta.default.mood_state_clusters
ORDER BY member_count DESC
```

---

### Q19: Listening Anomalies

**Purpose:** Detect unusual listening patterns
**Chart Type:** Table with Alert Icons

```sql
SELECT
    hour_of_day,
    anomaly_count,
    ROUND(avg_valence_deviation, 2) as mood_deviation,
    ROUND(avg_energy_deviation, 2) as energy_deviation,
    anomaly_severity,
    interpretation
FROM delta.default.listening_anomalies
ORDER BY
    CASE anomaly_severity
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
    END,
    anomaly_count DESC
```

---

### Q20: Sequential Pattern Analysis

**Purpose:** What mood sequences are common?
**Chart Type:** Sunburst Chart or Sankey Diagram

```sql
SELECT
    mood_category,
    pattern_type,
    occurrence_count,
    behavioral_insight
FROM delta.default.sequential_patterns
ORDER BY occurrence_count DESC
```

---

### Q21: Behavioral Segments Summary

**Purpose:** Overall listening archetype
**Chart Type:** Big Number with Description

```sql
SELECT
    listening_archetype,
    archetype_description,
    wellbeing_recommendation,
    total_listens,
    ROUND(overall_valence, 3) as avg_happiness,
    ROUND(overall_energy, 3) as avg_energy,
    ROUND(mood_variability, 3) as mood_variation,
    ROUND(energy_variability, 3) as energy_variation
FROM delta.default.behavioral_segments
```

---

## 🎯 Dashboard Templates

### Dashboard 1: Daily Overview

Combine these queries:
- Q1: Hourly Listening Activity (Line Chart)
- Q6: Weekend vs Weekday (Big Numbers)
- Q2: Top Artists (Bar Chart)
- Q9: Part of Day Analysis (Grouped Bar)

---

### Dashboard 2: Mood Analysis

Combine these queries:
- Q7: Mood Shift Throughout Day (Area Chart)
- Q10: Mood-Time Correlation (Scatter Plot)
- Q13: Mood Classification (Pie Chart)
- Q12: Actual vs Predicted Mood (Line Chart)

---

### Dashboard 3: Recommendations

Combine these queries:
- Q14: Mood Intervention Triggers (Table)
- Q15: Mood Improvement Recommendations (Table)
- Q16: Optimal Listening Times (Timeline)
- Q17: Personalized Playlists (Grouped Table)

---

### Dashboard 4: Deep Insights

Combine these queries:
- Q18: Mood State Clusters (Scatter Plot)
- Q20: Sequential Patterns (Sunburst)
- Q21: Behavioral Segment (Big Number)
- Q19: Listening Anomalies (Alert Table)

---

### Dashboard 5: Model Performance

Combine these queries:
- Q11: Model Performance Comparison (Table)
- Q12: Actual vs Predicted Mood (Line Chart)
- Custom: Training data size visualization
- Custom: Feature importance visualization

---

## 💡 Pro Tips

1. **Use Filters:** Add date filters to all queries for time-range analysis
   ```sql
   WHERE played_at >= CURRENT_DATE - INTERVAL '30' DAY
   ```

2. **Create Virtual Datasets:** Save complex queries as datasets for reuse

3. **Set Refresh Schedules:** Auto-refresh dashboards every 6 hours (match scheduler)

4. **Use Cache:** Enable result caching for faster dashboard loads

5. **Mobile-Friendly:** Test dashboards on mobile view for accessibility

6. **Annotations:** Add annotations to charts explaining insights

7. **Color Coding:**
   - Happy (high valence): Green/Yellow
   - Sad (low valence): Blue/Purple
   - High energy: Warm colors
   - Low energy: Cool colors

---

## 🚀 Getting Started Checklist

- [ ] Connect Superset to Trino (see TRINO_SUPERSET_SETUP.md)
- [ ] Test connection with simple query
- [ ] Create 5 basic datasets (one from each analytics type)
- [ ] Build your first chart (Q1: Hourly Listening Activity)
- [ ] Create your first dashboard (Daily Overview)
- [ ] Set up automatic refresh
- [ ] Share dashboard with team (if applicable)

---

**Need help?** Check Superset docs or Trino logs for troubleshooting.
