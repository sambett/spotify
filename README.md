# 🎵 Spotify Analytics Pipeline

A production-grade data pipeline for analyzing Spotify listening history and mental wellbeing patterns.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Spotify Developer Account ([Get one here](https://developer.spotify.com/dashboard))

### 1. Clone & Configure

```bash
cd C:\Users\SelmaB\Desktop\spotify

# Create .env file with your Spotify credentials
echo "CLIENT_ID=your_client_id_here" >> .env
echo "CLIENT_SECRET=your_client_secret_here" >> .env
echo "REDIRECT_URI=http://127.0.0.1:8888/callback" >> .env
```

### 2. First Run (Authentication)

```bash
# Build Docker image
docker-compose build

# Run pipeline once to authenticate
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

This opens a browser for Spotify OAuth, saves your token, and fetches your first data batch.

### 3. Start Automated Collection

```bash
# Start scheduler (runs every 6 hours automatically)
docker-compose up -d spotify-scheduler

# View logs
docker logs -f spotify-scheduler
```

**That's it!** Your pipeline now collects data every 6 hours automatically.

---

## 📊 What Data Gets Collected

### Bronze Layer (Raw Data)

```
data/bronze/
├── listening_history_bronze/              # Your Spotify plays
│   ├── track_id, played_at, track_name
│   ├── artist_name, album_name, duration_ms
│   └── Partitioned by date
│
├── my_tracks_features_bronze/             # Track metadata + audio features
│   ├── track_id, track_name, artist_name
│   ├── popularity, duration_ms, explicit
│   └── danceability, energy, valence, acousticness, etc.
│
├── my_tracks_features_bronze_synthetic/   # 🆕 Synthetic audio features
│   ├── Generated deterministically per track_id
│   ├── Used when real features unavailable (403 error)
│   └── source='synthetic' flag for provenance tracking
│
└── kaggle_tracks_bronze/                  # Reference catalog (114K tracks)
    ├── All fields from above
    └── track_genre included
```

### Audio Features for Mental Health Analysis

All tables include these mood-related features:

| Feature | Range | Mental Health Indicator |
|---------|-------|------------------------|
| `valence` | 0-1 | Musical positiveness (happiness) |
| `energy` | 0-1 | Intensity and activity level |
| `acousticness` | 0-1 | Acoustic vs electronic preference |
| `danceability` | 0-1 | Rhythmic engagement |
| `instrumentalness` | 0-1 | Vocal vs instrumental preference |
| `tempo` | 60-180 | Preferred pace (BPM) |

---

## 🆕 Synthetic Audio Features System

Due to Spotify API 403 errors on `/v1/audio-features`, the pipeline includes an intelligent fallback system:

### How It Works

1. **Deterministic Generation**: Hash-based seeding ensures reproducible features per track_id
2. **Automatic Population**: Runs after each ingestion to fill gaps
3. **Smart Preference**: Silver layer prefers real > synthetic > Kaggle
4. **Kill-Switch**: `ALLOW_SYNTHETIC=true/false` environment toggle

### Enable/Disable

```bash
# Enable synthetic features (default, recommended for development)
export ALLOW_SYNTHETIC=true
docker-compose up -d spotify-scheduler

# Disable synthetic features (when API is fixed)
export ALLOW_SYNTHETIC=false
docker-compose up -d spotify-scheduler
```

### Manual Generation

```bash
# Generate synthetic features for all tracks
docker-compose run --rm spotify-pipeline \
  python3 scripts/generate_synthetic_audio_features.py

# Populate only missing features
docker-compose run --rm spotify-pipeline \
  python3 scripts/populate_missing_features.py
```

**📖 Full Documentation**: See [SYNTHETIC_FEATURES_GUIDE.md](./SYNTHETIC_FEATURES_GUIDE.md)

---

## 🏗️ Architecture

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
│  • my_tracks_features_bronze_synthetic (🆕)                 │
│  • kaggle_tracks_bronze                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  SILVER LAYER (Enriched)                    │
├─────────────────────────────────────────────────────────────┤
│  • listening_with_features (joins all sources)              │
│  • Feature preference: real > synthetic > Kaggle            │
│  • Time dimensions: hour_of_day, part_of_day, is_weekend    │
│  • Data provenance: feature_source column                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   GOLD LAYER (Analytics)                    │
├─────────────────────────────────────────────────────────────┤
│  • mood_clusters (K-means on audio features)                │
│  • wellbeing_indicators (calculated metrics)                │
│  • track_recommendations (similarity-based)                 │
│  • temporal_patterns (time-based aggregations)              │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

- **Apache Spark 3.5.3**: Distributed data processing
- **Delta Lake**: ACID transactions, versioning, time travel
- **Docker**: Containerized deployment
- **Python 3.8**: ETL logic
- **Schedule**: Automated data collection every 6 hours

---

## 📁 Project Structure

```
spotify/
├── clients/              # API clients
│   ├── auth/             # Authentication logic
│   │   └── spotify_auth.py
│   └── spotify_api.py
├── config/               # Configuration
│   └── __init__.py
├── loaders/              # Data loaders (future)
├── mappers/              # Data transformers
│   ├── spotify_mapper.py
│   └── kaggle_mapper.py
├── schemas/              # Schema definitions
│   └── bronze_schemas.py
├── scripts/              # 🆕 Utility scripts
│   ├── generate_synthetic_audio_features.py
│   ├── populate_missing_features.py
│   └── build_silver_listening_with_features.py
├── utils/                # Utilities
│   └── logger.py
├── writers/              # Delta Lake writers
│   └── delta_writer.py
├── data/                 # Data storage
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── kaggle/
├── run_ingestion.py      # Main pipeline
├── scheduler.py          # Automated scheduler
├── docker-compose.yml    # Docker orchestration
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Common Operations

### View Data

```bash
# Enter container shell
docker exec -it spotify-scheduler /bin/bash

# Query Delta tables with PySpark
pyspark --packages io.delta:delta-core_2.12:2.4.0

# In PySpark:
df = spark.read.format('delta').load('/app/data/bronze/listening_history_bronze')
df.show(10)
df.printSchema()
```

### Rebuild Pipeline

```bash
# Stop scheduler
docker-compose down

# Rebuild with no cache
docker-compose build --no-cache

# Restart
docker-compose up -d spotify-scheduler
```

### Manual Ingestion

```bash
# One-time run
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

### Build Silver Layer

```bash
docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py \
  --history-path /app/data/bronze/listening_history_bronze \
  --real-features-path /app/data/bronze/my_tracks_features_bronze \
  --synthetic-features-path /app/data/bronze/my_tracks_features_bronze_synthetic \
  --kaggle-features-path /app/data/bronze/kaggle_tracks_bronze \
  --out-path /app/data/silver/listening_with_features
```

### Check Logs

```bash
# View scheduler logs
docker logs -f spotify-scheduler

# View last 100 lines
docker logs --tail 100 spotify-scheduler
```

---

## 🔧 Troubleshooting

### Issue: "403 Forbidden" on Audio Features

**Status**: Known issue with Spotify API permissions

**Solution**: Synthetic features system automatically fills the gap
- Set `ALLOW_SYNTHETIC=true` (default)
- Real features will be preferred when API is fixed
- See [FIX_403_AUDIO_FEATURES.md](./FIX_403_AUDIO_FEATURES.md) for permanent fix

### Issue: "No tracks found in listening history"

**Cause**: Need to listen to music first or re-authenticate

**Solution**:
```bash
# Delete old token
rm data/.spotify_tokens.json

# Re-run pipeline to re-authenticate
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

### Issue: "Kaggle dataset not found"

**Solution**: Download the dataset:
1. Get it from [Kaggle Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
2. Place CSV at: `data/kaggle/dataset.csv`
3. Re-run pipeline

### Issue: "Container keeps restarting"

**Check logs**:
```bash
docker logs --tail 50 spotify-scheduler
```

**Common causes**:
- Missing `.env` file
- Invalid Spotify credentials
- Port 8888 already in use

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](./README.md) | This file - overview and quick start |
| [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) | Detailed deployment instructions |
| [DATA_STATUS_EXPLAINED.md](./DATA_STATUS_EXPLAINED.md) | Current data status and structure |
| [FIX_403_AUDIO_FEATURES.md](./FIX_403_AUDIO_FEATURES.md) | Permanent fix for API 403 error |
| [SYNTHETIC_FEATURES_GUIDE.md](./SYNTHETIC_FEATURES_GUIDE.md) | 🆕 Complete synthetic features documentation |

---

## 🎓 Academic Project Context

This pipeline supports a **music-based mental wellbeing analysis** academic project:

### Research Questions
1. How do listening patterns correlate with mood throughout the day?
2. Can audio features predict mental wellbeing indicators?
3. What track characteristics are associated with positive vs negative moods?

### Analytics Types (5 Required)
1. **Descriptive**: Listening patterns by time/day
2. **Diagnostic**: Why certain moods occur at certain times
3. **Predictive**: Forecast mood based on listening behavior
4. **Prescriptive**: Recommend tracks to improve wellbeing
5. **Cognitive**: Cluster mood states using audio features

### Data Quality Note
Due to Spotify API limitations, synthetic features supplement real data for development.
Coverage: ~85% synthetic (testing), ~15% Kaggle (authentic). See [SYNTHETIC_FEATURES_GUIDE.md](./SYNTHETIC_FEATURES_GUIDE.md).

---

## 🚦 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Bronze Layer** | ✅ Complete | All 4 tables ingesting successfully |
| **Listening History** | ✅ Working | 1,000 events per run |
| **Track Features (Real)** | ⚠️ 403 Error | Spotify API permission issue |
| **Synthetic Features** | ✅ Ready | Automatic generation enabled |
| **Kaggle Dataset** | ✅ Loaded | 114,001 tracks |
| **Automated Scheduler** | ✅ Running | Every 6 hours |
| **Silver Layer** | ✅ Ready | Scripts available |
| **Gold Layer** | 🔄 Planned | Next phase |

---

## 🎯 Next Steps

### Immediate (Bronze Complete ✅)
- [x] Bronze ingestion working
- [x] Schemas match requirements
- [x] Automated scheduling setup
- [x] Synthetic features system implemented

### Next Phase - Silver Layer
- [ ] Run Silver layer builder
- [ ] Validate feature coverage
- [ ] Add data quality checks

### Next Phase - Gold Layer
- [ ] Implement mood clustering (K-means)
- [ ] Calculate wellbeing indicators
- [ ] Generate track recommendations
- [ ] Create temporal analytics

### Visualization
- [ ] Set up Apache Superset
- [ ] Create dashboards
- [ ] Implement 5 analytics types

---

## 🤝 Contributing

This is an academic project, but suggestions welcome:
1. Open an issue with your suggestion
2. Explain the use case
3. Provide example code if possible

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- **Spotify Web API**: For providing rich music data
- **Kaggle**: For the Spotify Tracks Dataset (114K tracks)
- **Apache Spark & Delta Lake**: For robust data processing
- **Docker**: For reproducible environments

---

## 📧 Contact

For questions about this academic project, please open an issue.

---

**Last Updated**: 2025-10-23
**Version**: 1.0.0 (Bronze Layer Complete + Synthetic Features)
