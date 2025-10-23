# 🔧 Fix for 403 Audio Features Error

## Current Situation

**What Works:**
- ✅ Fetching listening history (1,000 tracks)
- ✅ Getting track metadata (name, artist, popularity)

**What Doesn't Work:**
- ❌ Fetching audio features → **403 Forbidden**
- Result: All 47 unique tracks have **NULL** for danceability, energy, valence, etc.

## Why You Get 403

The `/v1/audio-features` endpoint has special requirements:

1. **It doesn't require user authentication** (it's public track data)
2. **Your user OAuth token might not have access** depending on app settings
3. **Some Spotify apps are restricted** from this endpoint by default

## ✅ SOLUTION 1: Use Client Credentials Flow (Recommended)

Audio features are **public data** - they don't need user permission. Use an **app-only token** instead of user token.

### Step 1: Create a new auth method for app-only access

Create `clients/auth/app_auth.py`:

```python
"""
App-only authentication for public Spotify endpoints.
Used for audio features which don't require user permission.
"""
import requests
import base64
from typing import Optional
from datetime import datetime, timedelta

class SpotifyAppAuth:
    """Client Credentials flow for public endpoints."""

    TOKEN_URL = "https://accounts.spotify.com/api/token"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

    def get_access_token(self) -> str:
        """Get app-only access token using Client Credentials."""

        # Check if current token is still valid
        if self.access_token and self.expires_at:
            if datetime.now() < self.expires_at - timedelta(minutes=5):
                return self.access_token

        # Request new token
        auth_str = f"{self.client_id}:{self.client_secret}"
        b64_auth_str = base64.b64encode(auth_str.encode()).decode()

        headers = {
            'Authorization': f'Basic {b64_auth_str}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {'grant_type': 'client_credentials'}

        response = requests.post(self.TOKEN_URL, headers=headers, data=data)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to get app token: {response.text}")

        token_data = response.json()
        self.access_token = token_data['access_token']
        expires_in = token_data.get('expires_in', 3600)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in)

        return self.access_token
```

### Step 2: Modify SpotifyAPIClient to use dual authentication

Update `clients/spotify_api.py`:

```python
from clients.auth.app_auth import SpotifyAppAuth

class SpotifyAPIClient:
    def __init__(self, config: AppConfig, auth_client: SpotifyAuthClient):
        self.config = config
        self.user_auth = auth_client  # For user endpoints
        self.app_auth = SpotifyAppAuth(  # For public endpoints
            config.spotify.client_id,
            config.spotify.client_secret
        )
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                     method: str = 'GET', use_app_token: bool = False) -> Dict:
        """Make authenticated API request."""

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                # Choose token type
                if use_app_token:
                    token = self.app_auth.get_access_token()
                else:
                    token = self.user_auth.get_valid_token()

                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }

                # ... rest of request logic

    def fetch_audio_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch audio features using app-only token."""

        if not track_ids:
            return []

        if len(track_ids) > 100:
            # Batch processing
            all_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                features = self.fetch_audio_features(batch)
                all_features.extend(features)
            return all_features

        params = {'ids': ','.join(track_ids)}

        logger.debug(f"Fetching audio features for {len(track_ids)} tracks")

        # USE APP TOKEN instead of user token
        data = self._make_request('/audio-features', params=params,
                                  use_app_token=True)  # ← KEY CHANGE
        features = data.get('audio_features', [])

        features = [f for f in features if f is not None]

        logger.debug(f"Fetched audio features for {len(features)} tracks")
        return features
```

### Step 3: Test it

```bash
docker-compose build
docker-compose run --rm spotify-pipeline python3 run_ingestion.py
```

**Expected Result:**
- ✅ Audio features should now fetch successfully
- ✅ Your 47 tracks will have danceability, energy, valence populated

---

## ✅ SOLUTION 2: Fix Spotify App Settings (Alternative)

If Client Credentials doesn't work, check your Spotify App Dashboard:

1. Go to https://developer.spotify.com/dashboard
2. Select your app
3. Click "Edit Settings"
4. Under "APIs used", verify these are enabled:
   - Web API
   - User Library Read/Write
5. Save changes
6. Delete token: `rm data/.spotify_tokens.json`
7. Re-run pipeline to re-authenticate

---

## ✅ SOLUTION 3: Use Kaggle Data (Current Workaround)

**This is what you're doing now** - it works but has limitations:

**Pros:**
- ✅ 114,000 tracks with full audio features
- ✅ Covers most popular music
- ✅ No API changes needed

**Cons:**
- ❌ Won't have features for obscure/new tracks
- ❌ Features might be slightly outdated
- ❌ Can't get real-time features

**Join Strategy:**
```python
# Silver layer transformation
listening = spark.read.format('delta').load('bronze/listening_history')
kaggle = spark.read.format('delta').load('bronze/kaggle_tracks')

enriched = listening.join(
    kaggle,
    on='track_id',
    how='left'  # Keep all listens, add features where available
)

# Check coverage
coverage = enriched.filter(enriched.valence.isNotNull()).count()
total = enriched.count()
print(f"Coverage: {coverage}/{total} = {coverage/total*100:.1f}%")
```

---

## 🎯 Recommendation

**For Maximum Data Quality:** Implement **Solution 1** (Client Credentials)

This will give you audio features for ALL your tracks, not just ones in Kaggle.

**For Quick Academic Project:** Keep using **Solution 3** (Kaggle)

114K tracks is more than enough for analysis.

---

## 📊 Expected Improvement

**Before Fix:**
```
Your 47 tracks: 0/47 with audio features (0%)
Kaggle coverage: ~70-80% of your tracks
```

**After Fix:**
```
Your 47 tracks: 47/47 with audio features (100%)
Complete coverage of ALL your listening history
```

---

## ⚠️ Important Note

Even with the fix, you **still benefit from having Kaggle data** because:
- Provides genre information (Spotify API doesn't)
- Offers 114K tracks for recommendations
- Acts as backup if API fails

**Best setup:** Both Spotify API features + Kaggle catalog! 🎯
