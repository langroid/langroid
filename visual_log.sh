#!/bin/bash

echo "Starting import profiling..."

# Log file name creation
unique_id=$(openssl rand -hex 8 2>/dev/null || head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
LOG_FILE="import_logs/langroid_import_${unique_id}.log"

echo "Generating log file: ${LOG_FILE}"

# Directory creation
mkdir -p import_logs
echo "Created/verified 'import_logs' directory."

# Create a temporary file for the Python code
TEMP_PY_FILE=$(mktemp)

# Check for command-line argument (Python script path)
if [ -n "$1" ]; then
  PYTHON_SCRIPT="$1"
  echo "Profiling imports for script: ${PYTHON_SCRIPT}"
else
  # Default import statement (if no script provided)
  echo "import langroid" > "$TEMP_PY_FILE"
  PYTHON_SCRIPT="$TEMP_PY_FILE"
  echo "No script provided. Profiling default 'import langroid' using temp file."
fi

# Langroid import (or script execution) with logging
echo "Profiling imports using Python's -X importtime..."
python -X importtime "$PYTHON_SCRIPT" >> "${LOG_FILE}" 2>&1
echo "Import profiling completed."

# Tuna check and execution
if python -c "import tuna" 2>/dev/null; then
  echo "Running tuna on ${LOG_FILE}..."
  tuna "${LOG_FILE}"
else
  echo "Error: The 'tuna' Python package is not installed."
  echo "Please install it using: pip install tuna"
  exit 1
fi

# Clean up the temporary file (if it was created)
if [ "$PYTHON_SCRIPT" = "$TEMP_PY_FILE" ]; then
  rm "$TEMP_PY_FILE"
  echo "Removed temporary file."
fi

echo "Import profiling process completed."