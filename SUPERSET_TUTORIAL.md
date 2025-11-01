# 📊 Superset Step-by-Step Visualization Tutorial

## 🎯 Goal
Create 5 powerful visualizations to understand your music listening patterns and ML insights in 15 minutes!

---

## ✅ Prerequisites Checklist

- [x] Superset running: http://localhost:8088
- [x] Trino connected: `trino://admin@trino:8080/delta/default`
- [x] Login credentials: admin / admin
- [x] Data pipeline has run at least once

---

## 📋 Tutorial Overview

We'll create:
1. **Mood Timeline** - How your mood changes throughout the day
2. **Music Personality** - Your behavioral segment
3. **Mood Boosters** - Personalized recommendations
4. **Energy Scatter** - Visual clustering of your music taste
5. **Model Performance** - ML accuracy metrics

---

## 🚀 STEP 1: Add Your First Dataset

### 1.1 Navigate to Datasets
- Click **Data** in the top menu
- Select **Datasets** from the dropdown
- Click **+ Dataset** (blue button, top right)

### 1.2 Configure Dataset
```
Database:     [Select] Spotify Analytics (your Trino connection)
Schema:       default
Table:        mood_predictions
```

### 1.3 Save
- Click **Add** button
- You should see "Dataset added successfully!"

### 1.4 Repeat for These Tables
Add these 4 more datasets (same process):
- `listening_with_features`
- `mood_improvement_recommendations`
- `behavioral_segments`
- `model_performance_metrics`

---

## 📈 CHART 1: Mood Timeline (5 minutes)

### What You'll See
A line chart showing how your predicted mood (valence) changes from morning to night!

### Steps

#### 1. Create Chart
- Go to **Charts** → **+ Chart**
- Choose dataset: `mood_predictions`
- Click the dataset name to proceed

#### 2. Select Chart Type
- Chart type: **Line Chart**
- Click to confirm

#### 3. Configure Chart

**Left Panel - Data Tab:**
```
Dimensions (X-Axis):
  - hour_of_day

Metrics (Y-Axis):
  - Click "+ Add Metric"
  - Choose "Simple"
  - Aggregate: AVG
  - Column: predicted_valence
  - Label: "Average Mood"
```

**Left Panel - Customize Tab:**
```
Chart Title: "My Mood Throughout the Day"
X Axis Label: "Hour of Day (0-23)"
Y Axis Label: "Mood Score (0=Sad, 1=Happy)"

Color Scheme: supersetColors
```

#### 4. Generate Chart
- Click **Update Chart** button (top right)
- You should see a line graph!

#### 5. Analyze
Look for patterns:
- Morning mood dip?
- Evening happiness peak?
- Afternoon energy slump?

#### 6. Save
- Click **Save** (top right)
- Chart name: "Mood Timeline"
- Add to: Create new dashboard "My Spotify Analytics"
- Click **Save & Go to Dashboard**

**✅ First chart complete!**

---

## 🧠 CHART 2: Music Personality (3 minutes)

### What You'll See
Your listening personality classification (e.g., "Balanced Explorer")

### Steps

#### 1. Create Chart
- **Charts** → **+ Chart**
- Dataset: `behavioral_segments`
- Chart type: **Big Number with Trendline**

#### 2. Configure

**Data Tab:**
```
Metric:
  - Click "Simple"
  - Column: segment_name
  - Label: "My Listening Personality"

Time Range: No filter (leave blank)
```

**Customize Tab:**
```
Header Text: "YOUR MUSIC PERSONALITY"
Subheader: interpretation  (from column)
```

#### 3. Update Chart
- Click **Update Chart**
- You should see your personality type!

#### 4. Save
- Save as "Music Personality"
- Add to dashboard: "My Spotify Analytics"

---

## 🎵 CHART 3: Mood-Boosting Playlist (5 minutes)

### What You'll See
ML-generated personalized recommendations to improve your mood!

### Steps

#### 1. Create Chart
- **Charts** → **+ Chart**
- Dataset: `mood_improvement_recommendations`
- Chart type: **Table**

#### 2. Configure

**Data Tab - Columns:**
Click "+ Add column" for each:
```
1. track_name          → Label: "Track"
2. artist_name         → Label: "Artist"
3. avg_valence         → Label: "Happiness Score"
4. avg_energy          → Label: "Energy"
5. recommendation_reason → Label: "Why Recommended"
```

**Query Tab:**
```
Row Limit: 10
Sort by: avg_valence (descending)
```

**Customize Tab:**
```
Table Title: "Your Mood-Boosting Playlist"
Show totals: No
```

#### 3. Style the Table
**Conditional Formatting** (optional):
- avg_valence > 0.7 → Green background
- avg_valence < 0.3 → Red background

#### 4. Update & Save
- Click **Update Chart**
- Save as "Mood Boosters"
- Add to dashboard

**You now have personalized recommendations!** 🎉

---

## 🎯 CHART 4: Music Taste Scatter (4 minutes)

### What You'll See
Visual clustering showing your music preferences (Energy vs Happiness)

### Steps

#### 1. Create Chart
- Dataset: `listening_with_features`
- Chart type: **Scatter Plot**

#### 2. Configure

**Data Tab:**
```
X Axis:
  - Column: energy
  - Label: "Energy Level"

Y Axis:
  - Aggregate: AVG
  - Column: valence
  - Label: "Happiness Score"

Bubble Size:
  - COUNT(*)
  - Label: "Play Count"

Limit: 1000 rows
```

**Customize Tab:**
```
Chart Title: "My Music Taste Map"
X Axis Label: "Energy (0=Calm, 1=Intense)"
Y Axis Label: "Happiness (0=Sad, 1=Happy)"

Color: #1f77b4 (blue)
Point Size: 10
```

#### 3. Update Chart
You should see clusters!
- Top-right quadrant: Happy & Energetic songs
- Bottom-left: Sad & Calm songs
- Analyze your preference!

#### 4. Save
- Save as "Music Taste Scatter"
- Add to dashboard

---

## 📊 CHART 5: ML Model Performance (3 minutes)

### What You'll See
How accurate your machine learning models are!

### Steps

#### 1. Create Chart
- Dataset: `model_performance_metrics`
- Chart type: **Table**

#### 2. Configure

**Columns:**
```
1. model_name
2. model_type
3. r2_score → Label: "R² Score"
4. rmse → Label: "RMSE"
5. accuracy → Label: "Accuracy"
6. f1_score → Label: "F1 Score"
```

**Customize:**
```
Title: "ML Model Performance Metrics"
Number Format for scores: .3f (3 decimals)
```

#### 3. Update & Save
- Save as "Model Performance"
- Add to dashboard

---

## 🎨 STEP 2: Arrange Your Dashboard

### 1. Open Dashboard
- Go to **Dashboards**
- Click "My Spotify Analytics"

### 2. Arrange Charts
Drag and drop to organize:

```
┌─────────────────────────────────────────┐
│     YOUR MUSIC PERSONALITY              │
│  [segment + interpretation + scores]    │
├──────────────────┬──────────────────────┤
│  Mood Timeline   │  Music Taste Scatter │
│  [line chart]    │  [scatter plot]      │
├──────────────────┴──────────────────────┤
│     Mood-Boosting Playlist               │
│     [table with recommendations]         │
├──────────────────────────────────────────┤
│     ML Model Performance                 │
│     [metrics table]                      │
└──────────────────────────────────────────┘
```

### 3. Resize Charts
- Drag corners to resize
- Make personality card wide but short
- Make playlist table full width

### 4. Save Dashboard
- Click **Save** (top right)

**🎉 You now have a complete analytics dashboard!**

---

## 🔥 BONUS CHARTS (If You Have Time)

### Bonus 1: Hourly Listening Activity
```
Dataset: listening_patterns_by_time
Chart: Bar Chart
X: hour_of_day
Y: play_count
```

### Bonus 2: Weekend vs Weekday Mood
```
Dataset: weekend_vs_weekday
Chart: Mixed Chart (Bar + Line)
Compare average valence and energy
```

### Bonus 3: Feature Correlations
```
Dataset: feature_correlations
Chart: Heatmap
Shows which audio features relate to each other
```

---

## 💡 Advanced: Custom SQL Queries

Want more control? Use **SQL Lab**!

### 1. Open SQL Lab
- Top menu → **SQL** → **SQL Lab**

### 2. Write Custom Query
Example - Your top sad songs:
```sql
SELECT
    track_name,
    artist_name,
    valence as sadness_score,
    COUNT(*) as play_count
FROM listening_with_features
WHERE valence < 0.3
GROUP BY track_name, artist_name, valence
ORDER BY play_count DESC
LIMIT 10
```

### 3. Run Query
- Click **Run** (or Ctrl+Enter)

### 4. Save as Dataset
- Click **Save** → **Save dataset**
- Name: "My Sad Songs"

### 5. Create Chart from It
- Now you can visualize your custom query!

---

## 🎓 More Query Examples

### Most Played Songs This Week
```sql
SELECT
    track_name,
    artist_name,
    COUNT(*) as plays
FROM listening_with_features
WHERE played_at > CURRENT_DATE - INTERVAL '7' DAY
GROUP BY track_name, artist_name
ORDER BY plays DESC
LIMIT 20
```

### Energy Progression Throughout Day
```sql
SELECT
    EXTRACT(HOUR FROM played_at) as hour,
    AVG(energy) as avg_energy,
    AVG(valence) as avg_mood,
    COUNT(*) as play_count
FROM listening_with_features
GROUP BY EXTRACT(HOUR FROM played_at)
ORDER BY hour
```

### Your Chill-Out Playlist
```sql
SELECT DISTINCT
    track_name,
    artist_name,
    energy,
    valence
FROM listening_with_features
WHERE energy < 0.4
  AND valence > 0.5
  AND acousticness > 0.5
ORDER BY valence DESC
LIMIT 30
```

---

## 🔧 Troubleshooting

### Chart shows "No data"
- Check if table has data: `SELECT COUNT(*) FROM table_name`
- Verify time range filters
- Check aggregation settings

### Connection timeout
- Restart Trino: `docker-compose restart trino`
- Check Trino is healthy: http://localhost:8080

### Chart looks wrong
- Check data types (numeric vs string)
- Verify aggregation (AVG, SUM, COUNT)
- Try simpler version first

### Can't find a table
- Go to **Data** → **Datasets**
- Refresh browser
- Check Trino connection

---

## 📱 Dashboard Features

### Filters
Add dashboard-level filters:
1. Edit dashboard
2. Click **+** (Add Filter)
3. Choose column (e.g., `hour_of_day`)
4. Apply to all charts

### Auto-Refresh
Set dashboard to auto-refresh:
1. Edit dashboard
2. **...** menu → **Set auto-refresh interval**
3. Choose interval (e.g., 5 minutes)

### Share Dashboard
1. Click **Share** button
2. Copy link
3. Or export as PDF/PNG

---

## 🎯 Key Insights to Look For

### In Mood Timeline:
- Are you happier in mornings or evenings?
- Any mid-day dips?
- Weekend patterns different?

### In Music Taste Scatter:
- Do you prefer high-energy happy songs?
- Or calm melancholic ones?
- Balanced taste?

### In Recommendations:
- Which tracks boost your mood most?
- Pattern in recommended genres?
- Artists you should listen to more?

### In Personality:
- What's your segment?
- Does it match your self-perception?
- Are the recommendations helpful?

---

## 🎉 You're Done!

You now have:
✅ Complete analytics dashboard
✅ ML-powered insights
✅ Personalized recommendations
✅ Understanding of your music taste

**Next Steps**:
1. Share your dashboard with friends
2. Create more custom charts
3. Explore the data with SQL Lab
4. Wait for next pipeline run (every 6 hours) to see new data

---

## 📚 Additional Resources

- **Full ML Documentation**: See `ML_ANALYTICS_GUIDE.md`
- **All Available Tables**: 22 tables documented
- **Pipeline Status**: `docker logs spotify-scheduler`
- **Trino Web UI**: http://localhost:8080

Happy visualizing! 🎵📊
