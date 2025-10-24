#!/bin/bash
# Spotify Analytics Pipeline - Manual Execution Script
# Run all stages sequentially with verification

set -e  # Exit on error

echo "========================================="
echo "Spotify Analytics Pipeline - Manual Run"
echo "========================================="
echo ""

# Function to run a stage and check for errors
run_stage() {
    local stage_name=$1
    local command=$2

    echo "========================================="
    echo "STAGE: $stage_name"
    echo "========================================="
    echo "Command: $command"
    echo ""

    if eval "$command"; then
        echo "✅ $stage_name COMPLETED"
    else
        echo "❌ $stage_name FAILED"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""
}

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "Docker is running ✅"
echo ""

# Ask user what to run
echo "What do you want to run?"
echo "  1) Complete pipeline (all stages)"
echo "  2) Bronze layer only (ingestion)"
echo "  3) Silver layer only"
echo "  4) Gold layers only (all 5 analytics types)"
echo "  5) ML models only"
echo "  6) Custom (pick stages)"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo "Running COMPLETE PIPELINE..."
        echo ""

        run_stage "Bronze: Ingestion (Spotify API + Kaggle)" \
            "docker-compose run --rm spotify-pipeline python3 run_ingestion.py"

        run_stage "Silver: Transformation" \
            "docker-compose run --rm spotify-pipeline python3 scripts/build_silver_listening_with_features.py"

        run_stage "Gold: Descriptive Analytics" \
            "docker-compose run --rm spotify-pipeline python3 gold/descriptive/build_descriptive_analytics.py"

        run_stage "Gold: Diagnostic Analytics" \
            "docker-compose run --rm spotify-pipeline python3 gold/diagnostic/build_diagnostic_analytics.py"

        run_stage "Gold: Prescriptive Analytics" \
            "docker-compose run --rm spotify-pipeline python3 gold/prescriptive/build_prescriptive_analytics.py"

        run_stage "Gold: Cognitive Analytics" \
            "docker-compose run --rm spotify-pipeline python3 gold/cognitive/build_cognitive_analytics.py"

        run_stage "Gold: Predictive Analytics + ML Models" \
            "docker-compose run --rm spotify-pipeline bash -c 'pip3 install --no-cache-dir -q numpy scikit-learn pandas && python3 gold/predictive/build_predictive_models.py'"

        echo "========================================="
        echo "✅ COMPLETE PIPELINE FINISHED"
        echo "========================================="
        echo ""
        echo "Verify results:"
        echo "  docker exec -it trino trino --catalog delta --schema default --execute 'SHOW TABLES'"
        ;;

    2)
        run_stage "Bronze: Ingestion" \
            "docker-compose run --rm spotify-pipeline python3 run_ingestion.py"
        ;;

    3)
        run_stage "Silver: Transformation" \
            "docker-compose run --rm spotify-pipeline python3 scripts/build_silver_listening_with_features.py"
        ;;

    4)
        echo "Running ALL GOLD LAYERS..."
        echo ""

        run_stage "Gold: Descriptive" \
            "docker-compose run --rm spotify-pipeline python3 gold/descriptive/build_descriptive_analytics.py"

        run_stage "Gold: Diagnostic" \
            "docker-compose run --rm spotify-pipeline python3 gold/diagnostic/build_diagnostic_analytics.py"

        run_stage "Gold: Prescriptive" \
            "docker-compose run --rm spotify-pipeline python3 gold/prescriptive/build_prescriptive_analytics.py"

        run_stage "Gold: Cognitive" \
            "docker-compose run --rm spotify-pipeline python3 gold/cognitive/build_cognitive_analytics.py"

        run_stage "Gold: Predictive + ML" \
            "docker-compose run --rm spotify-pipeline bash -c 'pip3 install --no-cache-dir -q numpy scikit-learn pandas && python3 gold/predictive/build_predictive_models.py'"
        ;;

    5)
        run_stage "ML Models Training" \
            "docker-compose run --rm spotify-pipeline bash -c 'pip3 install --no-cache-dir -q numpy scikit-learn pandas && python3 gold/predictive/build_predictive_models.py'"
        ;;

    6)
        echo "Custom stage selection not implemented yet."
        echo "Please edit this script or run stages manually."
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Pipeline execution completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Verify data in Trino: docker exec -it trino trino --catalog delta --schema default"
echo "  2. Access Superset: http://localhost:8088 (admin/admin)"
echo "  3. View logs: docker logs spotify-scheduler --tail 100"
echo ""
