"""
Master script to build all data layers: Silver → Gold (all analytics types)

This orchestrates the complete pipeline from Bronze to Gold.
"""
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: str, description: str, timeout: int = 600) -> bool:
    """Run a Python script and return success status."""
    logger.info("=" * 80)
    logger.info(f"RUNNING: {description}")
    logger.info("=" * 80)

    try:
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Log output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")

        if result.returncode != 0:
            logger.error(f"❌ {description} FAILED with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
            return False

        logger.info(f"✅ {description} COMPLETED")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} TIMED OUT after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"❌ {description} FAILED: {e}")
        return False


def main():
    """Build all layers in sequence."""
    logger.info("=" * 80)
    logger.info("🚀 SPOTIFY ANALYTICS - COMPLETE PIPELINE BUILD")
    logger.info("=" * 80)

    base_path = Path(__file__).parent.parent

    # Define pipeline stages
    stages = [
        # SILVER LAYER
        {
            'script': str(base_path / 'scripts' / 'build_silver_listening_with_features.py'),
            'description': 'Silver Layer: Listening with Features',
            'timeout': 300
        },

        # GOLD LAYER - DESCRIPTIVE ANALYTICS
        {
            'script': str(base_path / 'gold' / 'descriptive' / 'build_descriptive_analytics.py'),
            'description': 'Gold Layer: Descriptive Analytics',
            'timeout': 300
        },

        # GOLD LAYER - DIAGNOSTIC ANALYTICS
        {
            'script': str(base_path / 'gold' / 'diagnostic' / 'build_diagnostic_analytics.py'),
            'description': 'Gold Layer: Diagnostic Analytics',
            'timeout': 300
        },

        # GOLD LAYER - PREDICTIVE ANALYTICS
        {
            'script': str(base_path / 'gold' / 'predictive' / 'build_predictive_models.py'),
            'description': 'Gold Layer: Predictive Analytics',
            'timeout': 600
        },

        # GOLD LAYER - PRESCRIPTIVE ANALYTICS
        {
            'script': str(base_path / 'gold' / 'prescriptive' / 'build_prescriptive_analytics.py'),
            'description': 'Gold Layer: Prescriptive Analytics',
            'timeout': 300
        },

        # GOLD LAYER - COGNITIVE ANALYTICS
        {
            'script': str(base_path / 'gold' / 'cognitive' / 'build_cognitive_analytics.py'),
            'description': 'Gold Layer: Cognitive Analytics',
            'timeout': 600
        },
    ]

    # Run all stages
    results = []
    for stage in stages:
        script_path = Path(stage['script'])

        # Skip if script doesn't exist yet
        if not script_path.exists():
            logger.warning(f"⚠️  Script not found (skipping): {script_path.name}")
            results.append(False)
            continue

        success = run_script(
            stage['script'],
            stage['description'],
            stage['timeout']
        )
        results.append(success)

        # Stop on first failure (unless you want to continue)
        if not success:
            logger.error(f"Pipeline stopped due to failure in: {stage['description']}")
            break

    # Summary
    logger.info("=" * 80)
    logger.info("PIPELINE BUILD SUMMARY")
    logger.info("=" * 80)

    total = len(stages)
    successful = sum(results)
    failed = total - successful

    logger.info(f"Total stages: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if all(results):
        logger.info("=" * 80)
        logger.info("✅ ALL LAYERS BUILT SUCCESSFULLY!")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("❌ PIPELINE BUILD INCOMPLETE")
        logger.error("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
