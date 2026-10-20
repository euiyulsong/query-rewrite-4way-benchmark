#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo 'ERROR: export OPENAI_API_KEY before running.' >&2
  exit 1
fi
python prepare.py --n 100 --shots 3 --seed 42
python run.py --model gpt-6-luna --workers 5 --qrecc-mode proxy
printf '\nOpen results/report.html in your browser.\n'
