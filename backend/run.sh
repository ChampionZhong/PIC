#!/bin/bash
# Run uvicorn with limited watch directories to avoid file watch limit

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root to ensure proper imports
cd "$PROJECT_ROOT"

# Only watch the backend directory and exclude large directories
uvicorn backend.main:app \
    --reload \
    --reload-dir backend \
    --reload-include "*.py" \
    --host 0.0.0.0 \
    --port 8000

