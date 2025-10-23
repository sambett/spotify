#!/usr/bin/env python3
"""
Test script to verify Spotify Analytics Pipeline setup.
Run this before the main pipeline to catch issues early.
"""
import sys
import os
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    
    print("✅ Python version OK")
    return True


def check_imports():
    """Check all required imports."""
    print_header("Required Packages")
    
    packages = [
        ("pyspark", "PySpark"),
        ("delta", "Delta Lake"),
        ("requests", "Requests"),
        ("dotenv", "Python-dotenv"),
        ("cryptography", "Cryptography"),
    ]
    
    all_ok = True
    for module, name in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError as e:
            print(f"❌ {name}: NOT FOUND")
            print(f"   Install with: pip install {module}")
            all_ok = False
    
    return all_ok


def check_environment_variables():
    """Check required environment variables."""
    print_header("Environment Variables")
    
    # Load .env if not in Docker
    if not os.getenv('KUBERNETES_SERVICE_HOST'):  # Simple Docker check
        from dotenv import load_dotenv
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv()
            print(f"📄 Loaded from: {env_path.absolute()}")
        else:
            print("⚠️  .env file not found, using system environment")
    
    required_vars = {
        'CLIENT_ID': 'Spotify Client ID',
        'CLIENT_SECRET': 'Spotify Client Secret',
        'REDIRECT_URI': 'OAuth Redirect URI',
    }
    
    all_ok = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask secret values
            if 'SECRET' in var or 'PASSWORD' in var:
                display = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '***'
            else:
                display = value
            print(f"✅ {var}: {display}")
        else:
            print(f"❌ {var}: NOT SET")
            print(f"   Required for: {description}")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check required directories exist."""
    print_header("Directory Structure")
    
    required_dirs = [
        'data',
        'data/bronze',
        'data/kaggle',
        'data/logs',
        'data/.tokens',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ (will be created)")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"   ✓ Created")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                all_ok = False
    
    return all_ok


def check_kaggle_dataset():
    """Check if Kaggle dataset exists."""
    print_header("Kaggle Dataset")
    
    kaggle_path = Path('data/kaggle/dataset.csv')
    
    if kaggle_path.exists():
        size = kaggle_path.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"✅ Found: {kaggle_path}")
        print(f"   Size: {size_mb:.2f} MB")
        
        # Try to count lines
        try:
            with open(kaggle_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"   Lines: {lines:,}")
        except Exception as e:
            print(f"   Warning: Couldn't count lines: {e}")
        
        return True
    else:
        print(f"⚠️  Not found: {kaggle_path}")
        print("   Pipeline will run but won't load Kaggle data")
        print("   To add: Place CSV file at data/kaggle/dataset.csv")
        return True  # Not critical


def check_tokens():
    """Check if Spotify tokens exist."""
    print_header("Spotify Tokens")
    
    token_path = Path('data/.spotify_tokens.json')
    
    if token_path.exists():
        print(f"✅ Found: {token_path}")
        print("   Authentication should work without browser")
        
        # Try to load and validate
        try:
            import json
            with open(token_path, 'r') as f:
                tokens = json.load(f)
            
            has_access = 'access_token' in tokens
            has_refresh = 'refresh_token' in tokens
            
            print(f"   Access token: {'✓' if has_access else '✗'}")
            print(f"   Refresh token: {'✓' if has_refresh else '✗'}")
            
            if 'expires_at' in tokens:
                from datetime import datetime
                expires = datetime.fromtimestamp(tokens['expires_at'])
                now = datetime.now()
                if expires > now:
                    remaining = (expires - now).total_seconds() / 3600
                    print(f"   Expires in: {remaining:.1f} hours")
                else:
                    print("   Status: Expired (will be refreshed)")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not validate tokens: {e}")
        
        return True
    else:
        print(f"⚠️  Not found: {token_path}")
        print("   First run will require browser authentication")
        print("   Make sure you can access a browser or pre-generate tokens")
        return True  # Not critical for test


def check_spark():
    """Test Spark session creation."""
    print_header("Spark Session")
    
    try:
        from pyspark.sql import SparkSession
        
        print("Creating test Spark session...")
        spark = (SparkSession.builder
            .appName("PreflightTest")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())
        
        spark.sparkContext.setLogLevel("ERROR")
        
        # Test basic operations
        print(f"✅ Spark version: {spark.version}")
        
        # Test Delta
        test_data = [(1, "test")]
        df = spark.createDataFrame(test_data, ["id", "value"])
        print(f"✅ DataFrame creation works")
        
        spark.stop()
        print("✅ Spark session OK")
        return True
        
    except Exception as e:
        print(f"❌ Spark test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "🔍 " * 20)
    print("   SPOTIFY ANALYTICS PIPELINE - PREFLIGHT CHECK")
    print("🔍 " * 20)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("Environment Variables", check_environment_variables),
        ("Directory Structure", check_directories),
        ("Kaggle Dataset", check_kaggle_dataset),
        ("Spotify Tokens", check_tokens),
        ("Spark Session", check_spark),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results[name] = False
    
    # Summary
    print_header("Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready to run pipeline.")
        return 0
    elif passed >= total - 2:  # Allow 2 warnings
        print("\n⚠️  Some non-critical checks failed. Pipeline may still work.")
        return 0
    else:
        print("\n❌ Critical checks failed. Please fix issues before running pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
