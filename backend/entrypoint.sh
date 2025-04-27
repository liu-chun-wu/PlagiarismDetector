#!/bin/bash
# set -o allexport
# source .env 2>/dev/null
# set +o allexport

echo "✅ Checking environment variables..."

REQUIRED_VARS=("OPENAI_APIKEY" "BACKEND_PORT" \
"BACKEND_API_URL_TEXT_GENERATE" "BACKEND_API_URL_TEXT_REPHRASE" \
"BACKEND_API_URL_PDF_GENERATE" "BACKEND_API_URL_PDF_REPHRASE")

MISSING_VARS=()
for VAR in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!VAR}" ]; then
    MISSING_VARS+=("$VAR")
  fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
  echo "❌ ERROR: Missing required environment variables:"
  for VAR in "${MISSING_VARS[@]}"; do
    echo "  - $VAR"
  done
  exit 1
fi

echo "✅ Environment variables loaded successfully."

# 啟用 Conda 環境
echo "🔁 Activating Conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate webapp || { echo "❌ FAILED: Could not activate Conda environment."; exit 1; }
echo "✅ Conda environment activated."

# 啟動 Gunicorn，改成載入 flask_api.py 裡的 app
echo "🚀 Starting Gunicorn on port ${BACKEND_PORT}..."
exec gunicorn --workers 1 --bind 0.0.0.0:${BACKEND_PORT} flask_api:app --timeout 3600 --log-level debug

