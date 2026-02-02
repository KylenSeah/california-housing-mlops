#!/bin/bash
# 
# California Housing MLOps Pipeline Orchestrator
# -----------------------------------------------------------------------------
# This script serves as the standardized entry point for the pipeline.
# Defaults to training all models.
# 
# Usage:
# ./run.sh # Train all models
# ./run.sh train [models] # Train specific models
# ./run.sh predict --file test.csv # Run inference on a file
# ./run.sh predict --json '{"longitude": -122.23, ...}' # Run inference on JSON input
# ./run.sh --help # Show full CLI help
# -----------------------------------------------------------------------------
set -euo pipefail

echo "===================================================="
echo "AIAP Technical Assessment: California Housing Pipeline"
echo "===================================================="
echo "Default: Training all 3 models (LGBMRegressor, RandomForestRegressor, XGBRegressor)"
echo "Config: configs/config.yaml"
echo "Usage:  ./run.sh [command] [args]"
echo "  - Train:   ./run.sh train LGBMRegressor"
echo "  - Predict: ./run.sh predict --file data/test.csv"
echo "  - JSON:    ./run.sh predict --json '{\"longitude\": -122.2, ...}'"
echo "  - Note:    No arguments defaults to 'train all' "
echo "----------------------------------------------------"

# Set PYTHONPATH to make package discoverable (no install needed)
export PYTHONPATH=src

# Default to training all models if no arguments provided; otherwise pass args directly
if [ $# -eq 0 ]; then
  python -m california_housing.main train all
else
  python -m california_housing.main "$@"
fi

echo ""
echo "Pipeline finished."
echo "→ Metrics: artifacts/metrics/"
echo "→ Models: logged + registered in MLflow"
echo "→ Artifacts: artifacts/"
echo "===================================================="