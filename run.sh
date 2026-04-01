#!/bin/bash
# run.sh — Start the Breast Cancer Image Classification Based on Deep Transfer Learning Backend Server
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv"

echo "🔬 Breast Cancer Image Classification — Deep Transfer Learning"
echo "============================================================="

# Activate venv
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "❌ Virtual environment not found. Run setup first:"
  echo "   python3.11 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt"
  exit 1
fi

# Check if model exists — auto-generated on first server startup if missing
MODEL="$PROJECT_DIR/backend/saved_model/model_DenseNet121.keras"
if [ ! -f "$MODEL" ]; then
  echo ""
  echo "ℹ️  No local model found — the server will auto-generate one on first startup."
  echo "   Actual predictions use an analysis model downloaded automatically on first run."
  echo ""
fi

echo "🚀 Starting FastAPI server at http://localhost:8000"
echo "   API docs at http://localhost:8000/docs"
echo "   Press CTRL+C to stop"
echo ""

cd "$PROJECT_DIR/backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
