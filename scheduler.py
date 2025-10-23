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


def run_ingestion():
    """Execute the ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("Starting scheduled ingestion...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            ['python3', 'run_ingestion.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Ingestion completed successfully")
        else:
            logger.error(f"❌ Ingestion failed with code {result.returncode}")
            logger.error(result.stderr)

    except subprocess.TimeoutExpired:
        logger.error("❌ Ingestion timed out after 10 minutes")
    except Exception as e:
        logger.error(f"❌ Ingestion error: {e}")


def main():
    """Main scheduler loop."""
    logger.info("🕐 Spotify Analytics Scheduler Started")
    logger.info("=" * 60)
    logger.info("Schedule:")
    logger.info("  - Every 6 hours: Full data ingestion")
    logger.info("  - Maximizes data collection within Spotify API limits")
    logger.info("=" * 60)

    # Schedule ingestion every 6 hours (4 times per day)
    # This maximizes data collection without hitting rate limits
    schedule.every(6).hours.do(run_ingestion)

    # Run immediately on startup
    logger.info("Running initial ingestion...")
    run_ingestion()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
