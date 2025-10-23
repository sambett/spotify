"""
Pre-flight Check Script
Validates that all requirements are met before running the pipeline.
"""
import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required = {
        'dotenv': 'python-dotenv',
        'requests': 'requests',
        'cryptography': 'cryptography',
        'pyspark': 'pyspark',
        'delta': 'delta-spark'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def check_env_file():
    """Check .env file exists and has required variables"""
    print("\n🔍 Checking .env configuration...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("   ❌ .env file not found")
        return False
    
    print("   ✅ .env file exists")
    
    # Check required variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REDIRECT_URI']
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == '':
            print(f"   ❌ {var} - NOT SET")
            missing.append(var)
        else:
            # Mask the value for security
            masked = value[:8] + '...' if len(value) > 8 else '***'
            print(f"   ✅ {var} = {masked}")
    
    return len(missing) == 0


def check_directories():
    """Check required directories exist"""
    print("\n🔍 Checking directory structure...")
    
    required_dirs = [
        'data/bronze',
        'data/.tokens',
        'data/kaggle',
        'data/logs',
        'clients',
        'config',
        'loaders',
        'mappers',
        'writers',
        'schemas',
        'utils'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ⚠️  {dir_path}/ - creating...")
            path.mkdir(parents=True, exist_ok=True)
    
    return True


def check_kaggle_data():
    """Check if Kaggle dataset exists"""
    print("\n🔍 Checking Kaggle dataset...")
    
    kaggle_path = Path("data/kaggle/dataset.csv")
    if kaggle_path.exists():
        size_mb = kaggle_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Kaggle dataset found ({size_mb:.1f} MB)")
        return True
    else:
        print("   ⚠️  Kaggle dataset not found (optional)")
        print("   Place your Kaggle CSV at: data/kaggle/dataset.csv")
        return True  # Not critical


def check_java():
    """Check if Java is installed (required for Spark)"""
    print("\n🔍 Checking Java installation...")
    
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        # Java version info is in stderr
        version_info = result.stderr.split('\n')[0] if result.stderr else "Java installed"
        print(f"   ✅ {version_info}")
        return True
    except FileNotFoundError:
        print("   ⚠️  Java not found")
        print("   Spark requires Java 11 or 17")
        print("   Download from: https://adoptium.net/")
        return False
    except Exception as e:
        print(f"   ⚠️  Could not verify Java: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("🚀 PRE-FLIGHT CHECK - Spotify Analytics Pipeline")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Directories", check_directories),
        ("Kaggle Data", check_kaggle_data),
        ("Java", check_java)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n   ❌ Error checking {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    critical_passed = results[0][1] and results[1][1] and results[2][1]  # Python, deps, env
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run!")
        print("\nRun the pipeline:")
        print("  python run_ingestion.py")
        return 0
    elif critical_passed:
        print("⚠️  SOME CHECKS FAILED - May work with warnings")
        print("\nYou can try running:")
        print("  python run_ingestion.py")
        return 0
    else:
        print("❌ CRITICAL CHECKS FAILED - Fix issues before running")
        print("\nInstall dependencies:")
        print("  pip install pyspark delta-spark python-dotenv requests cryptography")
        return 1


if __name__ == "__main__":
    sys.exit(main())
