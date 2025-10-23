# 📊 Superset Dashboard Setup Guide

Complete guide to creating visualizations and dashboards in Apache Superset for your Spotify Analytics Pipeline.

---

## 🚀 Quick Start

### 1. Access Superset

Open your browser and go to: **http://localhost:8088**

**Login Credentials:**
- Username: `admin`
- Password: `admin`

*You'll be prompted to change the password on first login.*

---

## 🔌 Connect Data Source

Since Trino-Delta integration is complex, we have **two options**:

### Option A: Use PostgreSQL Upload (Recommended for Quick Start)

1. Export Gold tables to CSV
2. Upload to Superset
3. Create dashboards immediately

### Option B: Use Trino (Advanced)

Requires additional Trino configuration to properly mount Delta Lake volumes.

---

## 📤 Option A: Export Tables to CSV (Recommended)

### Step 1: Export Gold Tables

Run this script to export all Gold tables to CSV:

```bash
docker-compose run --rm spotify-pipeline python3 scripts/export_gold_to_csv.py
```

This creates CSV files in `data/exports/`:
- `mood_predictions.csv`
- `mood_time_correlations.csv`
- `listening_patterns_by_time.csv`
- And 11 more...

### Step 2: Upload to Superset

1. **Go to Data → Upload a CSV**
2. **Select CSV file** (e.g., `mood_predictions.csv`)
3. **Configure:**
   - Table Name: `mood_predictions`
   - Schema: Leave blank or use `public`
   - Database: `examples` (default Superset database)
4. **Click "Save"**

Repeat for all 14 Gold tables.

---

## 📊 Option B: Configure Trino Connection

### Step 1: Install Trino Driver in Superset

```bash
docker exec -it superset pip install trino
```

### Step 2: Add Database Connection

1. **Go to Data → Databases → + Database**
2. **Select "Trino"**
3. **Configure:**
   - Display Name: `Spotify Analytics`
   - SQLAlchemy URI: `trino://trino:8080/delta/default`
   - (No username/password needed)
4. **Test Connection**
5. **Click "Connect"**

### Step 3: Sync Tables

1. Go to **Data → Datasets**
2. Click **+ Dataset**
3. Select:
   - Database: `Spotify Analytics`
   - Schema: `default`
   - Table: Select from dropdown
4. Click **Add**

---

## 🎨 Create Dashboards

### Dashboard 1: Descriptive Analytics

**Purpose:** What happened in my listening history?

#### Charts to Create:

1. **Listening Patterns by Hour**
   - Dataset: `listening_patterns_by_time`
   - Chart: Line Chart
   - X-axis: `hour_of_day`
   - Y-axis: `total_plays`
   - Metric: SUM(total_plays)

2. **Top Tracks by Mood**
   - Dataset: `top_tracks_by_mood`
   - Chart: Bar Chart
   - X-axis: `track_name`
   - Y-axis: `play_count`
   - Group by: `mood_category`

3. **Audio Feature Distributions**
   - Dataset: `audio_feature_distributions`
   - Chart: Box Plot
   - Metrics: `mean`, `median`, `stddev`
   - Dimensions: `feature_name`

4. **Feature Source Coverage**
   - Dataset: `feature_source_coverage`
   - Chart: Pie Chart
   - Dimension: `feature_source`
   - Metric: `percentage`

5. **Temporal Trends**
   - Dataset: `temporal_trends`
   - Chart: Time Series
   - Time Column: `part_of_day`
   - Metrics: `avg_valence`, `avg_energy`

---

### Dashboard 2: Diagnostic Analytics

**Purpose:** Why do certain patterns occur?

#### Charts to Create:

1. **Mood vs Time Correlation Heatmap**
   - Dataset: `mood_time_correlations`
   - Chart: Heatmap
   - X-axis: `hour_of_day`
   - Y-axis: Feature (`avg_valence`, `avg_energy`)
   - Color: Intensity

2. **Feature Correlations Matrix**
   - Dataset: `feature_correlations`
   - Chart: Correlation Heatmap
   - Dimensions: `feature_1`, `feature_2`
   - Metric: `correlation`

3. **Weekend vs Weekday Comparison**
   - Dataset: `weekend_vs_weekday`
   - Chart: Grouped Bar Chart
   - X-axis: Metrics (`avg_valence`, `avg_energy`, `avg_danceability`)
   - Groups: `period` (Weekend/Weekday)

4. **Mood Shift Patterns**
   - Dataset: `mood_shift_patterns`
   - Chart: Stacked Area Chart
   - X-axis: `hour_of_day`
   - Y-axis: `shift_count`
   - Stack by: `shift_magnitude`

5. **Part of Day Drivers**
   - Dataset: `part_of_day_drivers`
   - Chart: Radar Chart
   - Dimensions: `part_of_day`
   - Metrics: All audio features

---

### Dashboard 3: Predictive Analytics

**Purpose:** What will happen based on patterns?

#### Charts to Create:

1. **Mood Prediction Accuracy**
   - Dataset: `mood_predictions`
   - Chart: Scatter Plot
   - X-axis: `valence` (actual)
   - Y-axis: `predicted_valence`
   - Color: `hour_of_day`
   - Add diagonal line (y=x) to show perfect prediction

2. **Energy Forecast**
   - Dataset: `energy_forecasts`
   - Chart: Dual Line Chart
   - X-axis: `hour_of_day`
   - Lines: `energy` (actual) vs `predicted_energy`

3. **Mood Classification Distribution**
   - Dataset: `mood_classifications`
   - Chart: Sankey Diagram
   - From: `mood_label` (actual)
   - To: `predicted_mood_label`

4. **Model Performance Comparison**
   - Dataset: `model_performance_metrics`
   - Chart: Bar Chart
   - X-axis: `model_name`
   - Y-axis Metrics: `rmse`, `mae`, `r2` (for regression)
   - Or: `accuracy`, `f1` (for classification)

5. **Prediction Confidence Intervals**
   - Dataset: `mood_predictions`
   - Chart: Line Chart with Confidence Bands
   - X-axis: `hour_of_day`
   - Y-axis: `predicted_valence`
   - Show prediction variance

---

## 🎯 Chart Creation Steps

### How to Create a Chart in Superset:

1. **Go to Charts → + Chart**
2. **Select:**
   - Dataset: (e.g., `mood_predictions`)
   - Chart Type: (e.g., Line Chart)
3. **Configure in "Data" tab:**
   - Metrics: What to measure (e.g., AVG(valence))
   - Dimensions: How to group (e.g., hour_of_day)
   - Filters: (Optional, e.g., WHERE energy > 0.5)
4. **Configure in "Customize" tab:**
   - Title, colors, labels
   - Legend position
   - Axis formatting
5. **Click "Update Chart"**
6. **Click "Save"** and add to dashboard

---

## 📋 Dashboard Assembly

### Creating a Dashboard:

1. **Go to Dashboards → + Dashboard**
2. **Name it:** (e.g., "Descriptive Analytics")
3. **Click "Edit Dashboard"**
4. **Drag charts** from right panel into dashboard grid
5. **Resize and arrange** charts
6. **Add Markdown components** for titles/descriptions
7. **Click "Save"**

---

## 🎨 Recommended Dashboard Layouts

### Descriptive Analytics Dashboard Layout:
```
┌─────────────────────────────────────────────┐
│  🎵 Spotify Listening Insights              │
│  Descriptive Analytics                      │
├───────────────────┬─────────────────────────┤
│ Listening Patterns│ Top Tracks by Mood      │
│ by Hour           │                         │
│ (Line Chart)      │ (Bar Chart)             │
├───────────────────┼─────────────────────────┤
│ Audio Feature     │ Feature Source Coverage │
│ Distributions     │ (Pie Chart)             │
│ (Box Plot)        │                         │
├───────────────────┴─────────────────────────┤
│ Temporal Trends (Time Series)               │
└─────────────────────────────────────────────┘
```

### Diagnostic Analytics Dashboard Layout:
```
┌─────────────────────────────────────────────┐
│  🔍 Why Patterns Occur                      │
│  Diagnostic Analytics                       │
├───────────────────┬─────────────────────────┤
│ Mood-Time         │ Feature Correlations    │
│ Correlations      │ Matrix                  │
│ (Heatmap)         │ (Heatmap)               │
├───────────────────┼─────────────────────────┤
│ Weekend vs Weekday│ Mood Shift Patterns     │
│ Comparison        │ (Stacked Area)          │
│ (Grouped Bar)     │                         │
└───────────────────┴─────────────────────────┘
```

### Predictive Analytics Dashboard Layout:
```
┌─────────────────────────────────────────────┐
│  🔮 Predictions & Forecasts                 │
│  Predictive Analytics                       │
├───────────────────┬─────────────────────────┤
│ Mood Prediction   │ Energy Forecast         │
│ Accuracy          │ (Dual Line)             │
│ (Scatter)         │                         │
├───────────────────┼─────────────────────────┤
│ Mood              │ Model Performance       │
│ Classification    │ Comparison              │
│ (Sankey)          │ (Bar Chart)             │
└───────────────────┴─────────────────────────┘
```

---

## 🎨 Color Scheme Recommendations

### Mood-Based Colors:
- **Happy (High Valence):** 🟢 Green, 🟡 Yellow
- **Sad (Low Valence):** 🔵 Blue, 🟣 Purple
- **Energetic (High Energy):** 🔴 Red, 🟠 Orange
- **Calm (Low Energy):** 🔵 Light Blue, 🟢 Teal

### Analytics Type Colors:
- **Descriptive:** Blue tones (#1f77b4)
- **Diagnostic:** Orange tones (#ff7f0e)
- **Predictive:** Green tones (#2ca02c)

---

## 📊 Sample SQL Queries for Custom Charts

### Complex Analysis Queries:

**1. Hourly Mood Trends**
```sql
SELECT
  hour_of_day,
  AVG(valence) as avg_happiness,
  AVG(energy) as avg_energy,
  COUNT(*) as plays
FROM listening_with_features
GROUP BY hour_of_day
ORDER BY hour_of_day
```

**2. Top Energy Tracks**
```sql
SELECT
  track_name,
  artist_name,
  AVG(energy) as avg_energy,
  AVG(valence) as avg_valence,
  COUNT(*) as play_count
FROM listening_with_features
GROUP BY track_name, artist_name
ORDER BY avg_energy DESC
LIMIT 20
```

**3. Mood Prediction Accuracy by Hour**
```sql
SELECT
  hour_of_day,
  AVG(ABS(valence - predicted_valence)) as avg_error,
  COUNT(*) as predictions
FROM mood_predictions
GROUP BY hour_of_day
ORDER BY hour_of_day
```

---

## 🚀 Quick Start Checklist

- [ ] Access Superset at http://localhost:8088
- [ ] Login with admin/admin
- [ ] Export Gold tables to CSV (or configure Trino)
- [ ] Upload/sync 14 Gold tables
- [ ] Create Descriptive Analytics dashboard (5 charts)
- [ ] Create Diagnostic Analytics dashboard (5 charts)
- [ ] Create Predictive Analytics dashboard (5 charts)
- [ ] Add filters for interactivity
- [ ] Share dashboards with stakeholders

---

## 💡 Pro Tips

1. **Use Filters:** Add global filters to dashboards (e.g., date range, mood category)
2. **Add Descriptions:** Use Markdown components to explain what each chart shows
3. **Cross-Filter:** Enable chart click-through to filter other charts
4. **Mobile View:** Test dashboard on mobile for responsive design
5. **Export:** Use "Download as Image" or "Download as PDF" for presentations
6. **Scheduled Reports:** Set up email reports to send dashboards automatically

---

## 🔧 Troubleshooting

### "No data returned"
- Check dataset has data: Go to SQL Lab and run `SELECT * FROM table_name LIMIT 10`
- Verify filters aren't too restrictive

### "Connection failed"
- Trino: Ensure Trino container is running (`docker ps`)
- CSV: Ensure CSV was uploaded successfully

### "Chart not updating"
- Clear Superset cache: Settings → List Dashboards → Clear Cache
- Refresh browser with Ctrl+F5

---

## 📚 Additional Resources

- [Superset Official Docs](https://superset.apache.org/docs/intro)
- [Trino Delta Lake Connector](https://trino.io/docs/current/connector/delta-lake.html)
- [Chart Types Guide](https://superset.apache.org/docs/using-superset/exploring-data)

---

**Next Steps:**
1. Follow Option A (CSV) for quick setup
2. Create your first dashboard
3. Iterate and refine visualizations
4. Share insights with your team!

Good luck with your dashboards! 🎉
