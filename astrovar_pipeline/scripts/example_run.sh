#!/usr/bin/env bash
set -euo pipefail
export GAIA_USER="your_username"
export GAIA_PASS="your_password"

astrovar run --config $(python -c "import pathlib; print(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'defaults.yaml')")
