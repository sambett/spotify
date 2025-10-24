"""
Complete end-to-end automated pipeline for Spotify Analytics.

This script orchestrates the entire data flow:
1. Bronze Layer: Ingest from Spotify API + Kaggle
2. Silver Layer: Transform and enrich data
3. Gold Layer: Build all 5 analytics types + ML models
4. (Optional) Sync to PostgreSQL for Superset

Designed to be called by scheduler.py every 6 hours.
"""
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_stage(script_path: str, description: str, timeout: int = 600, critical: bool = True) -> bool:
    """
    Run a pipeline stage and return success status.

    Args:
        script_path: Path to Python script
        description: Human-readable stage name
        timeout: Timeout in seconds
        critical: If True, pipeline stops on failure. If False, continues.

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"STAGE: {description}")
    logger.info("=" * 80)

    try:
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Log output (truncate if very long)
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-50:]:  # Last 50 lines only
                logger.info(f"  {line}")

        if result.returncode != 0:
            logger.error(f"❌ {description} FAILED with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")

            if critical:
                logger.error("🛑 Critical stage failed. Stopping pipeline.")
            else:
                logger.warning("⚠️  Non-critical stage failed. Continuing...")

            return False

        logger.info(f"✅ {description} COMPLETED")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} TIMED OUT after {timeout} seconds")
        if critical:
            logger.error("🛑 Critical stage timed out. Stopping pipeline.")
        return False

    except Exception as e:
        logger.error(f"❌ {description} FAILED: {e}")
        if critical:
            logger.error("🛑 Critical stage failed. Stopping pipeline.")
        return False


def main():
    """Execute complete automated pipeline."""
    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info("🚀 SPOTIFY ANALYTICS - COMPLETE AUTOMATED PIPELINE")
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Pipeline Stages:")
    logger.info("  1. Bronze Layer  - Ingest from Spotify API + Kaggle")
    logger.info("  2. Silver Layer  - Transform and enrich data")
    logger.info("  3. Gold Layer    - Build all 5 analytics types")
    logger.info("  4. ML Models     - Retrain predictive models")
    logger.info("  5. Sync          - (Optional) Sync to PostgreSQL")
    logger.info("=" * 80)
    logger.info("")

    base_path = Path(__file__).parent

    # Define pipeline stages
    stages = [
        # STAGE 1: BRONZE LAYER (CRITICAL)
        {
            'script': str(base_path / 'run_ingestion.py'),
            'description': '1. Bronze Layer Ingestion (Spotify API + Kaggle)',
            'timeout': 600,
            'critical': True  # Pipeline stops if this fails
        },

        # STAGE 2: SILVER LAYER (CRITICAL)
        {
            'script': str(base_path / 'scripts' / 'build_silver_listening_with_features.py'),
            'description': '2. Silver Layer Transformation (Enrich data)',
            'timeout': 300,
            'critical': True  # Pipeline stops if this fails
        },

        # STAGE 3: GOLD LAYER - DESCRIPTIVE (NON-CRITICAL)
        {
            'script': str(base_path / 'gold' / 'descriptive' / 'build_descriptive_analytics.py'),
            'description': '3a. Gold - Descriptive Analytics (What happened?)',
            'timeout': 300,
            'critical': False  # Continue even if fails
        },

        # STAGE 4: GOLD LAYER - DIAGNOSTIC (NON-CRITICAL)
        {
            'script': str(base_path / 'gold' / 'diagnostic' / 'build_diagnostic_analytics.py'),
            'description': '3b. Gold - Diagnostic Analytics (Why did it happen?)',
            'timeout': 300,
            'critical': False
        },

        # STAGE 5: GOLD LAYER - PREDICTIVE + ML (NON-CRITICAL)
        {
            'script': str(base_path / 'gold' / 'predictive' / 'build_predictive_models.py'),
            'description': '3c. Gold - Predictive Analytics + ML Models (What will happen?)',
            'timeout': 600,
            'critical': False
        },

        # STAGE 6: GOLD LAYER - PRESCRIPTIVE (NON-CRITICAL)
        {
            'script': str(base_path / 'gold' / 'prescriptive' / 'build_prescriptive_analytics.py'),
            'description': '3d. Gold - Prescriptive Analytics (What should we do?)',
            'timeout': 300,
            'critical': False
        },

        # STAGE 7: GOLD LAYER - COGNITIVE (NON-CRITICAL)
        {
            'script': str(base_path / 'gold' / 'cognitive' / 'build_cognitive_analytics.py'),
            'description': '3e. Gold - Cognitive Analytics (Complex patterns)',
            'timeout': 600,
            'critical': False
        },

        # STAGE 8: SYNC TO POSTGRESQL (OPTIONAL - ONLY IF USING POSTGRES)
        # Uncomment if you want to sync to PostgreSQL for Superset
        # {
        #     'script': str(base_path / 'scripts' / 'sync_gold_to_postgres.py'),
        #     'description': '4. Sync Gold Tables to PostgreSQL (for Superset)',
        #     'timeout': 300,
        #     'critical': False
        # },
    ]

    # Execute pipeline
    results = []
    for stage in stages:
        script_path = Path(stage['script'])

        # Skip if script doesn't exist
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found, skipping: {script_path.name}")
            results.append(False)
            continue

        success = run_stage(
            stage['script'],
            stage['description'],
            stage['timeout'],
            stage['critical']
        )

        results.append(success)

        # Stop on critical failure
        if not success and stage['critical']:
            logger.error(f"🛑 Pipeline stopped due to critical failure in: {stage['description']}")
            break

    # Calculate summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    total = len([s for s in stages if Path(s['script']).exists()])
    successful = sum(results)
    failed = total - successful

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"⏱️  Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"📈 Stages Executed: {total}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info("=" * 80)

    # Overall status
    critical_failed = any(
        not results[i] and stages[i]['critical']
        for i in range(len(results))
        if i < len(stages) and Path(stages[i]['script']).exists()
    )

    if critical_failed:
        logger.error("❌ PIPELINE FAILED - Critical stage(s) failed")
        logger.error("=" * 80)
        return 1
    elif failed > 0:
        logger.warning("⚠️  PIPELINE COMPLETED WITH WARNINGS - Some non-critical stages failed")
        logger.warning("=" * 80)
        return 0  # Still return 0 since critical stages passed
    else:
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY - All stages passed!")
        logger.info("=" * 80)
        return 0


if __name__ == "__main__":
    sys.exit(main())
