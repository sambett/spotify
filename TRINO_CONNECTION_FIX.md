# Trino Connection Fix for Superset

## ❌ The Error

```
error 401: b'Basic authentication or X-Trino-Original-User or X-Trino-User must be sent'
```

This means Trino requires authentication headers that weren't included in the connection string.

---

## ✅ The Solution

Trino requires a **username** in the connection string, even if no password is needed.

### Correct Connection String:

```
trino://admin@trino:8080/delta/default
```

**Key Changes:**
- Added `admin@` before `trino:8080`
- Format: `trino://username@host:port/catalog/schema`

---

## 📋 Step-by-Step Connection in Superset

### 1. Access Superset
- URL: http://localhost:8088
- Login: `admin` / `admin`

### 2. Add Database Connection

1. Click **Settings** (gear icon, top right)
2. Click **Database Connections**
3. Click **+ Database** button

### 3. Choose Trino

Select **Trino** from the database list.

### 4. Enter Connection Details

**Display Name:**
```
Spotify Analytics (Trino)
```

**SQLAlchemy URI:**
```
trino://admin@trino:8080/delta/default
```

**Important Notes:**
- Username: `admin` (can be any string, Trino doesn't validate by default)
- No password needed
- Host: `trino` (Docker service name)
- Port: `8080`
- Catalog: `delta`
- Schema: `default`

### 5. Advanced Settings (Optional)

Click **Advanced** tab and configure:

- ✅ **Expose database in SQL Lab** - Check this
- ✅ **Allow CREATE TABLE AS** - Check this
- ✅ **Allow CREATE VIEW AS** - Check this
- ✅ **Allow DML** - Leave unchecked (read-only recommended)

**SQL Lab Settings:**
- Query timeout: `300` seconds

### 6. Test Connection

Click **Test Connection** button at the bottom.

You should see: ✅ **Connection looks good!**

### 7. Save

Click **Connect** to save the connection.

---

## 🔍 Alternative Connection Strings

If `admin` doesn't work, try these:

### Option 1: With user header
```
trino://trino@trino:8080/delta/default
```

### Option 2: With localhost (if inside Superset container)
```
trino://admin@trino:8080/delta/default
```

### Option 3: With explicit parameters
```
trino://admin@trino:8080/delta/default?http_scheme=http&auth=None
```

---

## 🧪 Test Your Connection

Once connected, go to **SQL Lab** → **SQL Editor** and run:

```sql
SHOW TABLES;
```

You should see all 22 Gold layer tables:

**Descriptive:**
- listening_patterns_by_time
- top_tracks_by_mood
- temporal_trends
- audio_feature_distributions
- feature_source_coverage

**Diagnostic:**
- mood_time_correlations
- feature_correlations
- weekend_vs_weekday
- mood_shift_patterns
- part_of_day_drivers

**Predictive:**
- mood_predictions
- energy_forecasts
- mood_classifications
- model_performance_metrics

**Prescriptive:**
- mood_improvement_recommendations
- optimal_listening_times
- personalized_playlist_suggestions
- mood_intervention_triggers

**Cognitive:**
- mood_state_clusters
- listening_anomalies
- sequential_patterns
- behavioral_segments

---

## 🐛 Troubleshooting

### Still Getting 401 Error?

**Check Trino is running:**
```bash
docker ps | grep trino
```

**Check Trino logs:**
```bash
docker logs trino --tail 50
```

**Restart Trino:**
```bash
docker-compose restart trino
```

**Wait 30 seconds for Trino to be ready:**
```bash
docker logs trino --follow
# Wait for "SERVER STARTED" message
```

### Connection Refused?

**Check network connectivity from Superset:**
```bash
docker exec superset curl http://trino:8080/v1/info
```

Should return JSON with Trino version info.

### Wrong Catalog or Schema?

**List available catalogs:**
```sql
SHOW CATALOGS;
```

**List schemas in delta catalog:**
```sql
SHOW SCHEMAS FROM delta;
```

---

## 🎯 Quick Test Query

Once connected, run this to verify your data:

```sql
SELECT
    COUNT(*) as total_plays,
    MIN(played_at) as first_play,
    MAX(played_at) as last_play,
    COUNT(DISTINCT artist_name) as unique_artists,
    COUNT(DISTINCT track_name) as unique_tracks
FROM delta.default.listening_with_features;
```

Expected output:
- total_plays: 1,504
- unique_artists: ~50-100
- unique_tracks: ~100-300

---

## 📚 Next Steps After Connection

1. **Create your first dataset:**
   - Go to **Data** → **Datasets**
   - Click **+ Dataset**
   - Select database, schema, table
   - Click **Add**

2. **Run example queries from `SUPERSET_QUERIES.md`:**
   - 21 ready-to-use queries
   - Covers all 5 analytics types

3. **Build your first dashboard:**
   - Use "Daily Overview" template (Q1, Q6, Q2, Q9)

---

## ✅ Confirmation Checklist

- [ ] Superset is running (http://localhost:8088)
- [ ] Trino is healthy (`docker ps | grep trino`)
- [ ] Connection string includes username: `trino://admin@trino:8080/delta/default`
- [ ] Test connection shows green checkmark
- [ ] `SHOW TABLES` returns 22+ tables
- [ ] Sample query returns data

---

**Questions?** Check Trino logs and Superset logs for detailed error messages.

```bash
docker logs trino --tail 100
docker logs superset --tail 100
```
