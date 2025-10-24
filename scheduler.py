"""
Automated scheduler for Spotify data ingestion.
Runs the pipeline at scheduled intervals to maximize data collection.
"""
import schedule
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Execute the complete automated pipeline (Bronze → Silver → Gold)."""
    logger.info("=" * 60)
    logger.info("Starting scheduled pipeline execution...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            ['python3', 'run_full_pipeline.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 60 minute timeout for full pipeline
        )

        if result.returncode == 0:
            logger.info("✅ Full pipeline completed successfully")
            logger.info("   Bronze → Silver → Gold → ML Models")
        else:
            logger.error(f"❌ Pipeline failed with code {result.returncode}")
            if result.stderr:
                logger.error(result.stderr[-500:])  # Last 500 chars

    except subprocess.TimeoutExpired:
        logger.error("❌ Pipeline timed out after 60 minutes")
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")


def main():
    """Main scheduler loop."""
    logger.info("🕐 Spotify Analytics Scheduler Started")
    logger.info("=" * 60)
    logger.info("Schedule:")
    logger.info("  - Every 6 hours: COMPLETE AUTOMATED PIPELINE")
    logger.info("    └─ Bronze: Spotify API + Kaggle ingestion")
    logger.info("    └─ Silver: Data transformation and enrichment")
    logger.info("    └─ Gold: All 5 analytics types + ML models")
    logger.info("  - Maximizes data collection within Spotify API limits")
    logger.info("=" * 60)

    # Schedule FULL PIPELINE every 6 hours (4 times per day)
    # This ensures all layers are kept up-to-date automatically
    schedule.every(6).hours.do(run_full_pipeline)

    # Run immediately on startup
    logger.info("Running initial pipeline execution...")
    run_full_pipeline()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
