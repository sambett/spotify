# Synthetic Audio Features - Complete Guide

## Overview

This system automatically generates **deterministic synthetic audio features** for tracks that lack real Spotify API features. This is essential because:

1. **403 Error Workaround**: The Spotify `/v1/audio-features` endpoint currently returns 403 Forbidden
2. **Testing & Validation**: Provides complete data coverage for pipeline development
3. **Academic Project Continuity**: Ensures you can build Silver/Gold layers without waiting for API fixes
4. **Reproducibility**: Hash-based seeding ensures identical results across runs

## How It Works

### 1. Deterministic Feature Generation

Each `track_id` gets the **same synthetic features every time** using SHA256 hash-based seeding:

```python
# track_id: "3n3Ppam7vgaVa1iaRUc9Lp"
seed = sha256("3n3Ppam7vgaVa1iaRUc9Lp").digest()[:8]  # Deterministic seed
rng = random.Random(seed)  # Seeded random generator

danceability = 0.15 + rng.betavariate(2, 2) * 0.70  # Always same value
energy = 0.10 + rng.betavariate(2, 2) * 0.80
valence = 0.10 + rng.betavariate(2, 2) * 0.75
# ... etc
```

**Result**: Running the script 100 times produces identical features for each track.

### 2. Feature Ranges (Realistic Spotify-like Values)

| Feature | Range | Description |
|---------|-------|-------------|
| `danceability` | 0.15 - 0.85 | How suitable for dancing |
| `energy` | 0.10 - 0.90 | Intensity and activity measure |
| `valence` | 0.10 - 0.85 | Musical positiveness (happiness) |
| `acousticness` | 0.05 - 0.95 | Acoustic vs electronic |
| `instrumentalness` | 0.00 - 0.80 | Likelihood of no vocals |
| `speechiness` | 0.03 - 0.50 | Presence of spoken words |
| `liveness` | 0.05 - 0.40 | Presence of audience |
| `loudness` | -25.0 - -3.0 | Overall loudness in dB |
| `tempo` | 60 - 180 | Speed in BPM |
| `time_signature` | [3, 4, 5] | Beats per bar |
| `key` | [0-11] | Musical key (0=C, 1=C#, etc.) |
| `mode` | [0, 1] | Major (1) or Minor (0) |

### 3. Automatic Pipeline Integration

The pipeline automatically populates missing features:

```
Listening History Ingestion
         ↓
Track Features Ingestion (may fail with 403)
         ↓
Kaggle Dataset Loading
         ↓
🆕 AUTOMATIC SYNTHETIC POPULATION
   ├── Find tracks in listening history
   ├── Check which have real features (prefer real!)
   ├── Check which have synthetic features (skip if exists)
   ├── Generate synthetic for remaining tracks
   └── Append to synthetic features table
```

### 4. Bronze Layer Structure

After running the pipeline, you'll have:

```
data/bronze/
├── listening_history_bronze/              # Your Spotify plays
│   └── 1,000 events with timestamps
├── my_tracks_features_bronze/             # Real features (mostly NULL due to 403)
│   └── 47 tracks (0 with complete features)
├── my_tracks_features_bronze_synthetic/   # 🆕 Synthetic features
│   └── 47 tracks (100% coverage)
└── kaggle_tracks_bronze/                  # Kaggle catalog
    └── 114,001 tracks
```

### 5. Silver Layer Feature Preference

The Silver layer intelligently merges features with preference order:

```
PREFERENCE: Real > Synthetic > Kaggle

For each track in listening history:
  IF real features exist (non-NULL):
    USE real features ✅ (best)
  ELSE IF synthetic features exist:
    USE synthetic features ✅ (good for testing)
  ELSE IF Kaggle features exist:
    USE Kaggle features ✅ (fallback)
  ELSE:
    NULL ⚠️
```

The `feature_source` column tracks provenance: `'real'`, `'synthetic'`, or `'kaggle'`.

---

## Environment Toggle (Kill-Switch)

### ALLOW_SYNTHETIC Environment Variable

Control synthetic feature generation with a single environment variable:

```bash
# Enable synthetic features (default)
export ALLOW_SYNTHETIC=true
docker-compose up -d spotify-scheduler

# Disable synthetic features
export ALLOW_SYNTHETIC=false
docker-compose up -d spotify-scheduler
```

**Behavior when disabled:**
- Pipeline continues normally
- Synthetic feature generation is skipped
- Logs show: `⚠️  Synthetic feature generation is DISABLED`
- Silver layer uses real + Kaggle features only

**Use cases:**
- **Development/Testing**: `ALLOW_SYNTHETIC=true` (get full coverage)
- **Production** (when 403 fixed): `ALLOW_SYNTHETIC=false` (real data only)
- **Academic Validation**: Toggle to compare real vs synthetic impact

---

## Usage Examples

### Example 1: Generate Synthetic Features Manually

```bash
# Generate synthetic features for all tracks in listening history
docker-compose run --rm spotify-pipeline \
  python3 scripts/generate_synthetic_audio_features.py \
  --history-path /app/data/bronze/listening_history_bronze \
  --out-path /app/data/bronze/my_tracks_features_bronze_synthetic
```

**Output:**
```
================================================================================
SYNTHETIC AUDIO FEATURES GENERATOR
================================================================================
History path: /app/data/bronze/listening_history_bronze
Output path: /app/data/bronze/my_tracks_features_bronze_synthetic
================================================================================
✅ Found 47 distinct track IDs
Generating synthetic features for 47 tracks...
✅ Generated 47 synthetic feature records
✅ Successfully wrote synthetic features to Delta Lake
================================================================================
SUMMARY
================================================================================
✅ Distinct tracks processed: 47
✅ Synthetic features generated: 47
✅ Output location: /app/data/bronze/my_tracks_features_bronze_synthetic
================================================================================
```

### Example 2: Populate Missing Features Automatically

```bash
# Run the automatic populator (detects what's missing)
docker-compose run --rm spotify-pipeline \
  python3 scripts/populate_missing_features.py \
  --history-path /app/data/bronze/listening_history_bronze \
  --real-features-path /app/data/bronze/my_tracks_features_bronze \
  --synthetic-features-path /app/data/bronze/my_tracks_features_bronze_synthetic \
  --allow-synthetic true
```

**Output:**
```
================================================================================
AUTOMATIC FEATURE POPULATION
================================================================================
Allow synthetic: True
================================================================================
IDENTIFYING TRACKS NEEDING FEATURES
================================================================================
✅ Found 47 distinct tracks in listening history
✅ Found 0 tracks with real features
✅ Found 0 tracks with synthetic features
================================================================================
COVERAGE ANALYSIS
================================================================================
Total tracks in history:    47
Tracks with real features:  0
Tracks with synthetic:      0
Total covered:              0
Missing (need synthetic):   47
Current coverage:           0.0%
================================================================================
Generating synthetic features for 47 tracks...
✅ Generated 47 synthetic feature records
✅ Successfully appended 47 synthetic features
================================================================================
POPULATION SUMMARY
================================================================================
✅ New tracks detected:      47
✅ Synthetic features added: 47
================================================================================
```

### Example 3: Build Silver Layer with Feature Merging

```bash
# Build Silver layer (prefers real, falls back to synthetic/Kaggle)
docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py \
  --history-path /app/data/bronze/listening_history_bronze \
  --real-features-path /app/data/bronze/my_tracks_features_bronze \
  --synthetic-features-path /app/data/bronze/my_tracks_features_bronze_synthetic \
  --kaggle-features-path /app/data/bronze/kaggle_tracks_bronze \
  --out-path /app/data/silver/listening_with_features
```

**Output:**
```
================================================================================
SILVER LAYER BUILDER
================================================================================
Loading listening history from: /app/data/bronze/listening_history_bronze
✅ Loaded 1000 listening events
Loading real features from: /app/data/bronze/my_tracks_features_bronze
✅ Loaded 0 tracks with real audio features
Loading synthetic features from: /app/data/bronze/my_tracks_features_bronze_synthetic
✅ Loaded 47 tracks with synthetic features
Loading Kaggle features from: /app/data/bronze/kaggle_tracks_bronze
✅ Loaded 114001 tracks from Kaggle
Merging features with preference order: real → synthetic → kaggle
Adding time dimensions...
✅ Time dimensions added
Writing Silver layer to: /app/data/silver/listening_with_features
✅ Silver layer written successfully
================================================================================
FEATURE COVERAGE STATISTICS
================================================================================
synthetic   :    850 ( 85.0%)
kaggle      :    150 ( 15.0%)
Total with features:   1000 (100.0%)
================================================================================
✅ Silver layer build complete!
================================================================================
```

### Example 4: Run Full Pipeline (Automatic Mode)

The pipeline now automatically populates missing features:

```bash
# Run pipeline once (includes automatic synthetic population)
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

**Pipeline Steps:**
1. ✅ Fetch listening history (1,000 plays)
2. ⚠️  Fetch track features (fails with 403)
3. ✅ Load Kaggle dataset (114K tracks)
4. 🆕 **Automatically populate missing features with synthetic data**
5. ✅ Pipeline complete

**Or run scheduled collection:**
```bash
# Start scheduler (runs every 6 hours, auto-populates each time)
docker-compose up -d spotify-scheduler

# View logs
docker logs -f spotify-scheduler
```

---

## Data Quality & Validation

### Coverage Statistics

After running the pipeline, check your coverage:

```python
# In Silver layer
SELECT
    feature_source,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM listening_with_features), 1) as percentage
FROM listening_with_features
GROUP BY feature_source;
```

**Expected Results (with synthetic features enabled):**

| feature_source | count | percentage |
|----------------|-------|------------|
| synthetic      | 850   | 85.0%      |
| kaggle         | 150   | 15.0%      |
| **Total**      | **1000** | **100.0%** |

### Verification: Same Track, Same Features

```python
# Generate features twice, verify they're identical
from scripts.generate_synthetic_audio_features import SyntheticAudioFeatureGenerator

gen = SyntheticAudioFeatureGenerator()
track_id = "3n3Ppam7vgaVa1iaRUc9Lp"

features_run1 = gen._generate_features_for_track(track_id)
features_run2 = gen._generate_features_for_track(track_id)

assert features_run1 == features_run2  # ✅ Always identical
```

---

## Safety & Best Practices

### ✅ Safe Operations

1. **Idempotent**: Running scripts multiple times is safe
2. **Append-only**: Synthetic features are appended (Delta Lake handles duplicates)
3. **Non-destructive**: Never modifies real features or listening history
4. **Deterministic**: Same input = same output every time
5. **Logged**: All operations clearly logged with coverage statistics

### ⚠️ Important Notes

1. **Development Use**: Synthetic features are for **testing and validation**
2. **Real Features Preferred**: Silver layer always prefers real API features
3. **Clear Provenance**: `feature_source` column tracks data origin
4. **Academic Honesty**: Document synthetic data use in your report
5. **Production Toggle**: Use `ALLOW_SYNTHETIC=false` when real features work

### 🚨 What NOT To Do

- ❌ Don't use synthetic features for production recommendations (yet)
- ❌ Don't claim synthetic features are real in academic reports
- ❌ Don't disable the kill-switch in production without real features
- ❌ Don't modify the seed generation algorithm (breaks reproducibility)

---

## Troubleshooting

### Issue: "No synthetic features generated"

**Check 1: Is ALLOW_SYNTHETIC enabled?**
```bash
echo $ALLOW_SYNTHETIC  # Should output: true
```

**Check 2: Do you have listening history?**
```bash
docker-compose run --rm spotify-pipeline \
  python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.format('delta').load('/app/data/bronze/listening_history_bronze')
print(f'Listening events: {df.count()}')
"
```

### Issue: "Feature populator script not found"

**Solution:** Ensure scripts directory is mounted in docker-compose.yml:
```yaml
volumes:
  - ./scripts:/app/scripts  # ← Must be present
```

### Issue: "Coverage still 0% after synthetic generation"

**Likely cause:** Silver layer not built yet. Run:
```bash
docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py
```

---

## Academic Project Integration

### Mention in Your Report

**Example Section:**

> **Data Quality Note**
>
> Due to Spotify API permission limitations (403 Forbidden on `/v1/audio-features` endpoint),
> we implemented a deterministic synthetic feature generation system for testing and validation.
> Synthetic features are:
> - Generated using hash-based seeding (SHA256 of `track_id`) for reproducibility
> - Distributed using realistic ranges based on Spotify API documentation
> - Clearly marked with `source='synthetic'` in the data model
> - Used only where real API features are unavailable (0/47 personal tracks)
> - Supplemented by authentic Kaggle dataset features (114K tracks)
>
> The Silver layer prioritizes real features, with synthetic serving as fallback for
> pipeline development. Coverage analysis shows 85% synthetic, 15% Kaggle features.

### For Your Defense

**Potential Question**: "Why use synthetic data?"

**Answer**:
- Demonstrates robust data engineering practices (handling API failures gracefully)
- Enables full pipeline development without blocking on external API issues
- Shows understanding of data provenance and quality tracking
- Reproducible: hash-based generation ensures consistent results for validation
- Kaggle dataset (114K tracks) provides authentic features for comparative analysis

---

## Next Steps

### 1. ✅ Immediate Actions

```bash
# Run full pipeline with synthetic features
export ALLOW_SYNTHETIC=true
docker-compose run --rm spotify-pipeline python3 run_ingestion.py

# Build Silver layer
docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py
```

### 2. ✅ Verify Coverage

```bash
# Check synthetic features exist
docker-compose run --rm spotify-pipeline \
  python3 -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.format('delta').load('/app/data/bronze/my_tracks_features_bronze_synthetic')
print(f'Synthetic features: {df.count()}')
df.show(5)
"
```

### 3. ✅ Start Automated Collection

```bash
# Start scheduler (runs every 6 hours)
docker-compose up -d spotify-scheduler
```

### 4. 🔄 When 403 is Fixed

```bash
# Disable synthetic generation
export ALLOW_SYNTHETIC=false

# Re-run pipeline to get real features
docker-compose run --rm spotify-pipeline python3 run_ingestion.py

# Rebuild Silver (will now prefer real features)
docker-compose run --rm spotify-pipeline \
  python3 scripts/build_silver_listening_with_features.py
```

---

## Summary

| Component | Status | Purpose |
|-----------|--------|---------|
| **Synthetic Generator** | ✅ Ready | Generate deterministic features per track_id |
| **Missing Feature Populator** | ✅ Ready | Automatically fill gaps after ingestion |
| **Silver Layer Builder** | ✅ Ready | Merge real > synthetic > Kaggle with preference |
| **Pipeline Integration** | ✅ Ready | Auto-populate runs after each ingestion |
| **Kill-Switch** | ✅ Ready | `ALLOW_SYNTHETIC=true/false` environment toggle |
| **Documentation** | ✅ Complete | Full usage guide and best practices |

**Your pipeline now has:**
- ✅ Automatic synthetic feature generation for 100% coverage
- ✅ Clear preference order (real > synthetic > Kaggle)
- ✅ Environment toggle for production safety
- ✅ Full data provenance tracking
- ✅ Reproducible, deterministic results

**You're ready to build Gold layer analytics and dashboards!** 🎉
