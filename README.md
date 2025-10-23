# 🎵 Spotify Analytics Pipeline

A production-grade data analytics pipeline for Spotify listening behavior with advanced mood-based analysis using Apache Spark, Delta Lake, Trino, and Apache Superset.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Spotify Developer Account ([Get one here](https://developer.spotify.com/dashboard))
- Kaggle Dataset ([Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset))

### 1. Clone & Configure

```bash
# Clone repository
git clone <your-repo-url>
cd spotify

# Create .env file with your Spotify credentials
echo "CLIENT_ID=your_client_id_here" >> .env
echo "CLIENT_SECRET=your_client_secret_here" >> .env
echo "REDIRECT_URI=http://127.0.0.1:8888/callback" >> .env
echo "ALLOW_SYNTHETIC=true" >> .env
```

### 2. Add Kaggle Dataset

1. Download from [Kaggle Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
2. Place CSV at: `data/kaggle/dataset.csv`

### 3. First Run (Authentication)

```bash
# Build Docker image
docker-compose build spotify-pipeline

# Run pipeline once to authenticate
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

This opens a browser for Spotify OAuth, saves your token, and fetches your first data batch.

### 4. Start All Services

```bash
# Start all services (scheduler, Trino, Superset)
docker-compose up -d

# View logs
docker-compose logs -f
```

### 5. Access Services

- **Superset**: http://localhost:8088 (admin/admin)
- **Trino**: http://localhost:8080 (no auth required)
- **Scheduler**: Running in background (collects data every 6 hours)

**That's it!** Your complete analytics stack is now running.

---

## 📊 Architecture Overview

### Medallion Architecture (Bronze → Silver → Gold)

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
├─────────────────────────────────────────────────────────────┤
│  Spotify API          Kaggle Dataset    Synthetic Generator │
│  (listening history)  (114K tracks)     (deterministic)     │
└────────┬──────────────────┬───────────────────┬─────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER (Raw)                       │
├─────────────────────────────────────────────────────────────┤
│  • listening_history_bronze                                 │
│  • my_tracks_features_bronze                                │
│  • my_tracks_features_bronze_synthetic                      │
│  • kaggle_tracks_bronze                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  SILVER LAYER (Enriched)                    │
├─────────────────────────────────────────────────────────────┤
│  • listening_with_features                                  │
│    - Joins all sources (real > synthetic > Kaggle)          │
│    - Time dimensions: hour_of_day, part_of_day, is_weekend  │
│    - Data provenance: feature_source column                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   GOLD LAYER (Analytics)                    │
├─────────────────────────────────────────────────────────────┤
│  DESCRIPTIVE (What happened?)                               │
│  • listening_patterns_by_time                               │
│  • top_tracks_by_mood                                       │
│  • temporal_trends                                          │
│  • audio_feature_distributions                              │
│  • feature_source_coverage                                  │
│                                                             │
│  DIAGNOSTIC (Why did it happen?)                            │
│  • mood_time_correlations                                   │
│  • feature_correlations                                     │
│  • weekend_vs_weekday                                       │
│  • mood_shift_patterns                                      │
│  • part_of_day_drivers                                      │
│                                                             │
│  PREDICTIVE (What will happen?)                             │
│  • mood_predictions (R² = 0.84)                             │
│  • energy_forecasts (R² = 0.08)                             │
│  • mood_classifications (Accuracy = 97.8%)                  │
│  • model_performance_metrics                                │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                  QUERY & VISUALIZATION                      │
├─────────────────────────────────────────────────────────────┤
│  Trino (Federated SQL)  →  Apache Superset (Dashboards)    │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Data Processing** | Apache Spark | 3.5.3 | Distributed computing |
| **Data Lake** | Delta Lake | 3.2.1 | ACID transactions, versioning |
| **Query Engine** | Trino | 435 | Federated SQL queries |
| **Visualization** | Apache Superset | 3.0.0 | Business intelligence dashboards |
| **ML Framework** | Spark MLlib | 3.5.3 | Machine learning models |
| **Database** | PostgreSQL | 15 | Superset metadata |
| **Cache** | Redis | 7 | Superset caching |
| **Container Orchestration** | Docker Compose | - | Service management |
| **Language** | Python | 3.8 | Pipeline logic |

---

## 📁 Project Structure

```
spotify/
├── clients/                  # API clients
│   ├── auth/                # Spotify OAuth
│   └── spotify_api.py       # API wrapper
├── config/                   # Configuration
├── gold/                     # Gold layer analytics
│   ├── descriptive/         # What happened?
│   ├── diagnostic/          # Why did it happen?
│   └── predictive/          # What will happen?
├── ingestion/               # Data ingestion logic
├── loaders/                 # Data loaders
├── mappers/                 # Data transformers
├── schemas/                 # Schema definitions
├── scripts/                 # Utility scripts
│   ├── generate_synthetic_audio_features.py
│   ├── populate_missing_features.py
│   ├── build_silver_listening_with_features.py
│   └── ensure_ml_deps.sh
├── superset/                # Superset configuration
├── trino/                   # Trino catalog configuration
│   └── catalog/
│       └── delta.properties
├── utils/                   # Utilities
├── validators/              # Data validators
├── writers/                 # Delta Lake writers
├── data/                    # Data storage (not in git)
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── kaggle/
├── docker-compose.yml       # Service orchestration
├── Dockerfile               # Main container
├── entrypoint.sh           # Container startup script
├── requirements.txt         # Python dependencies
├── run_ingestion.py        # Main pipeline
└── scheduler.py            # Automated scheduler
```

---

## 🎯 5 Types of Analytics Implemented

### 1. Descriptive Analytics (What happened?)

**5 Tables Created:**
- `listening_patterns_by_time` - Aggregated listening by hour/day
- `top_tracks_by_mood` - Most played tracks by mood category
- `temporal_trends` - Trends over time periods
- `audio_feature_distributions` - Feature distributions (mean, median, stddev)
- `feature_source_coverage` - Data provenance tracking (real vs synthetic vs Kaggle)

**Run:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 gold/descriptive/build_descriptive_analytics.py
```

### 2. Diagnostic Analytics (Why did it happen?)

**5 Tables Created:**
- `mood_time_correlations` - Why certain moods occur at certain hours
- `feature_correlations` - Why features relate to each other
- `weekend_vs_weekday` - Why listening differs by day type
- `mood_shift_patterns` - Why mood changes throughout day
- `part_of_day_drivers` - What drives mood in morning/afternoon/evening/night

**Run:**
```bash
docker-compose run --rm spotify-pipeline \
  python3 gold/diagnostic/build_diagnostic_analytics.py
```

### 3. Predictive Analytics (What will happen?)

**3 ML Models + Metrics Table:**
- `mood_predictions` - Random Forest Regressor (RMSE: 0.0615, R²: 0.84)
- `energy_forecasts` - Linear Regression (RMSE: 0.1904, R²: 0.08)
- `mood_classifications` - Random Forest Classifier (Accuracy: 97.8%, F1: 0.98)
- `model_performance_metrics` - Model evaluation metrics

**Run:**
```bash
docker-compose run --rm spotify-pipeline bash -c \
  "pip3 install --no-cache-dir -q numpy==1.24.4 scikit-learn==1.3.2 pandas==2.0.3 matplotlib==3.7.5 seaborn==0.13.0 && \
   python3 gold/predictive/build_predictive_models.py"
```

**Anti-Overfitting Techniques Applied:**
- Data leakage prevention (excluded target features from predictors)
- Reduced model complexity (numTrees=20, maxDepth=5)
- Minimum samples per leaf (minInstancesPerNode=5)
- 80/20 train-test split with fixed seed

### 4. Prescriptive Analytics (What should we do?)

**Status**: Planned for next phase
- Track recommendations for mood improvement
- Optimal listening time suggestions
- Playlist generation for wellbeing

### 5. Cognitive Analytics (Complex pattern recognition)

**Status**: Planned for next phase
- K-means clustering for mood states
- Anomaly detection in listening patterns
- Deep learning for sequential pattern prediction

---

## 📊 Data Schema

### Audio Features for Mental Health Analysis

All tables include these mood-related features:

| Feature | Range | Mental Health Indicator |
|---------|-------|------------------------|
| `valence` | 0-1 | Musical positiveness (happiness) |
| `energy` | 0-1 | Intensity and activity level |
| `acousticness` | 0-1 | Acoustic vs electronic preference |
| `danceability` | 0-1 | Rhythmic engagement |
| `instrumentalness` | 0-1 | Vocal vs instrumental preference |
| `speechiness` | 0-1 | Presence of spoken words |
| `loudness` | -60-0 dB | Volume preference |
| `tempo` | 60-180 BPM | Preferred pace |

### Synthetic Audio Features System

Due to Spotify API 403 errors on `/v1/audio-features`, the pipeline includes an intelligent fallback:

**How It Works:**
1. **Deterministic Generation**: Hash-based seeding ensures reproducible features per track_id
2. **Automatic Population**: Runs after each ingestion to fill gaps
3. **Smart Preference**: Silver layer prefers real > synthetic > Kaggle
4. **Data Provenance**: `feature_source` column tracks origin

**Enable/Disable:**
```bash
# Enable synthetic features (default)
export ALLOW_SYNTHETIC=true

# Disable synthetic features
export ALLOW_SYNTHETIC=false
```

---

## 🛠️ Common Operations

### Build Complete Pipeline

```bash
# Build all layers (Bronze → Silver → Gold)
docker-compose run --rm spotify-pipeline python3 run_ingestion.py

docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py

docker-compose run --rm spotify-pipeline \
  python3 gold/descriptive/build_descriptive_analytics.py

docker-compose run --rm spotify-pipeline \
  python3 gold/diagnostic/build_diagnostic_analytics.py

docker-compose run --rm spotify-pipeline bash -c \
  "pip3 install --no-cache-dir -q numpy==1.24.4 scikit-learn==1.3.2 pandas==2.0.3 && \
   python3 gold/predictive/build_predictive_models.py"
```

### Query with Trino

```bash
# Connect to Trino
docker exec -it trino trino

# Query Delta tables
USE delta.default;
SHOW TABLES;

SELECT * FROM listening_with_features LIMIT 10;
SELECT * FROM mood_predictions LIMIT 10;
```

### Query with PySpark

```bash
# Enter container
docker exec -it spotify-scheduler /bin/bash

# Start PySpark with Delta Lake
pyspark --packages io.delta:delta-spark_2.12:3.2.1

# In PySpark:
from delta.tables import DeltaTable

df = spark.read.format('delta').load('/app/data/silver/listening_with_features')
df.show(10)
df.printSchema()

# Query Gold layer
mood_df = spark.read.format('delta').load('/app/data/gold/predictive/mood_predictions')
mood_df.show(10)
```

### Superset Configuration

1. **Access Superset**: http://localhost:8088
2. **Login**: admin/admin (change on first login)
3. **Add Database Connection**:
   - Database: Trino
   - SQLAlchemy URI: `trino://trino:8080/delta`
   - Test Connection
4. **Create Datasets** from tables in `delta.default` schema
5. **Build Dashboards**

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker logs -f spotify-scheduler
docker logs -f superset
docker logs -f trino
```

### Stop/Restart Services

```bash
# Stop all
docker-compose down

# Start all
docker-compose up -d

# Restart specific service
docker-compose restart superset
```

---

## 🔧 Troubleshooting

### Issue: "403 Forbidden" on Audio Features

**Status**: Known issue with Spotify API permissions

**Solution**: Synthetic features system automatically fills the gap
- Set `ALLOW_SYNTHETIC=true` in `.env` (default)
- Real features will be preferred when API is fixed

### Issue: "Superset not starting"

**Solution**:
```bash
# Remove old volumes and restart
docker-compose down
docker volume rm spotify_postgres-data
docker-compose up -d postgres redis superset
```

### Issue: "Trino connection refused"

**Solution**:
```bash
# Check Trino is healthy
docker ps --filter "name=trino"

# Restart Trino
docker-compose restart trino
```

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution**: Install ML dependencies before running predictive models:
```bash
docker-compose run --rm spotify-pipeline bash -c \
  "pip3 install --no-cache-dir -q numpy==1.24.4 scikit-learn==1.3.2 pandas==2.0.3 && \
   python3 gold/predictive/build_predictive_models.py"
```

### Issue: "Container keeps restarting"

**Check logs**:
```bash
docker logs --tail 50 <container-name>
```

**Common causes**:
- Missing `.env` file
- Invalid Spotify credentials
- Port conflicts (8080, 8088, 8888)

---

## 🚦 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Bronze Layer** | ✅ Complete | All 4 tables ingesting successfully |
| **Silver Layer** | ✅ Complete | 1,504 records with 100% feature coverage |
| **Gold - Descriptive** | ✅ Complete | 5 tables created |
| **Gold - Diagnostic** | ✅ Complete | 5 tables created |
| **Gold - Predictive** | ✅ Complete | 3 ML models + metrics (no overfitting) |
| **Gold - Prescriptive** | 🔄 Planned | Next phase |
| **Gold - Cognitive** | 🔄 Planned | Next phase |
| **Trino** | ✅ Running | Port 8080, healthy |
| **Superset** | ✅ Running | Port 8088, healthy |
| **Automated Scheduler** | ✅ Running | Every 6 hours |

---

## 📈 Model Performance

### Mood Prediction Model (Random Forest Regressor)
- **Target**: Valence (happiness level)
- **Features**: hour_of_day, day_of_week, energy, tempo, danceability, is_weekend
- **RMSE**: 0.0615
- **MAE**: 0.0464
- **R²**: 0.8355 (excellent predictive power)

### Energy Forecast Model (Linear Regression)
- **Target**: Energy level
- **Features**: hour_of_day, day_of_week, tempo, danceability, loudness
- **RMSE**: 0.1904
- **MAE**: 0.1563
- **R²**: 0.0808 (baseline performance)

### Mood Category Classifier (Random Forest Classifier)
- **Target**: Mood categories (Happy_Energetic, Happy_Calm, Sad_Energetic, Sad_Calm, Neutral)
- **Features**: hour_of_day, day_of_week, tempo, danceability, acousticness, loudness, speechiness
- **Accuracy**: 97.83%
- **F1 Score**: 0.9781
- **Note**: Data leakage prevented by excluding valence/energy from features

---

## 🎓 Academic Project Context

This pipeline supports a **music-based mental wellbeing analysis** academic project.

### Research Questions
1. How do listening patterns correlate with mood throughout the day?
2. Can audio features predict mental wellbeing indicators?
3. What track characteristics are associated with positive vs negative moods?

### 5 Analytics Types Implementation

| Type | Purpose | Status |
|------|---------|--------|
| **Descriptive** | What happened? | ✅ 5 tables |
| **Diagnostic** | Why did it happen? | ✅ 5 tables |
| **Predictive** | What will happen? | ✅ 3 models |
| **Prescriptive** | What should we do? | 🔄 Planned |
| **Cognitive** | Complex patterns | 🔄 Planned |

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](./README.md) | This file - complete overview |
| [COMPLETE_ARCHITECTURE_GUIDE.md](./COMPLETE_ARCHITECTURE_GUIDE.md) | Architecture details |
| [SYNTHETIC_FEATURES_GUIDE.md](./SYNTHETIC_FEATURES_GUIDE.md) | Synthetic features documentation |

---

## 🤝 Contributing

This is an academic project, but suggestions welcome:
1. Open an issue with your suggestion
2. Explain the use case
3. Provide example code if possible

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- **Spotify Web API**: For providing rich music data
- **Kaggle**: For the Spotify Tracks Dataset (114K tracks)
- **Apache Spark & Delta Lake**: For robust data processing
- **Trino**: For federated SQL queries
- **Apache Superset**: For beautiful data visualization
- **Docker**: For reproducible environments

---

## 📧 Contact

For questions about this academic project, please open an issue.

---

**Last Updated**: 2025-10-23
**Version**: 2.0.0 (Complete Pipeline: Bronze + Silver + Gold [Descriptive, Diagnostic, Predictive])
