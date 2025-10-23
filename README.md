# 🎵 Spotify Analytics Pipeline - Complete Setup

## 🎯 Quick Start (3 Steps - 10 Minutes)

### 1. Install Dependencies
```bash
cd C:\Users\SelmaB\Desktop\spotify
pip install pyspark delta-spark python-dotenv requests cryptography
```

### 2. Verify Setup
```bash
python preflight_check.py
```

### 3. Run First Time (Authenticate)
```bash
python run_ingestion.py
```
Opens browser for Spotify OAuth → Saves token → Fetches your data!

---

## ⏰ Schedule for 11 PM Daily

**PowerShell (as Administrator):**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\schedule_ingestion.ps1
```

**Test immediately:**
```powershell
Start-ScheduledTask -TaskName "SpotifyDataIngestion"
```

---

## 📊 What Gets Fetched

### 1. Listening History
- Recently played tracks (up to 1,000)
- Play timestamps
- Track/artist/album names

### 2. Audio Features (for Mood Analysis!)
- `valence` (0-1) - Musical happiness
- `energy` (0-1) - Intensity level
- `danceability` (0-1) - Dance suitability
- `acousticness`, `tempo`, `loudness`, etc.

### 3. Kaggle Dataset (Optional)
- ~100K tracks for recommendations
- Place CSV at: `data/kaggle/dataset.csv`

---

## 📁 Project Structure

```
spotify/
├── run_ingestion.py          # 🚀 Main script
├── preflight_check.py         # ✅ Validation
├── schedule_ingestion.ps1     # ⏰ Scheduler
├── run_manual.bat            # 🎯 Quick test
│
├── clients/                   # API clients
│   ├── auth/                 # OAuth & tokens
│   └── spotify_api.py        # Spotify wrapper
│
├── config/                    # Configuration
│   └── settings.py           # Env-based config
│
├── loaders/                   # Data loaders
├── mappers/                   # Schema mappers
├── writers/                   # Delta Lake writers
├── schemas/                   # Data schemas
└── utils/                     # Utilities
```

---

## 🔍 Checking Results

```bash
# View logs
Get-Content data\logs\ingestion_*.log -Tail 50

# Query data (Python)
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("View").getOrCreate()
df = spark.read.format("delta").load("data/bronze/listening_history_bronze")
df.show()
```

---

## 🐛 Troubleshooting

**"Module not found: pyspark"**
```bash
pip install pyspark delta-spark
```

**"Java not found"**
- Download Java 11/17: https://adoptium.net/

**"Spotify authentication failed"**
1. Check CLIENT_ID/CLIENT_SECRET in `.env`
2. Verify REDIRECT_URI: `http://127.0.0.1:8888/callback`
3. Run manually: `python run_ingestion.py`

**"No data fetched"**
- Ensure recent Spotify listening history
- Check Spotify privacy settings

---

## ✅ Project Status

**Complete (70%):**
- ✅ Bronze layer (data ingestion)
- ✅ Automated scheduling
- ✅ Audio features extraction

**To Build:**
- ⏳ Silver layer (transformations)
- ⏳ Gold layer (analytics)
- ⏳ ML models (mood clustering)

**See PROJECT_ASSESSMENT.md for detailed roadmap.**

---

## 🚀 You're Ready!

```bash
# 1. Validate
python preflight_check.py

# 2. Test
python run_ingestion.py

# 3. Schedule
.\schedule_ingestion.ps1
```

Done! Pipeline runs at 11 PM daily. 🎉
