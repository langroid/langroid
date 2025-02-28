#!/bin/bash

echo "Starting import profiling..."

# Log file name creation
unique_id=$(openssl rand -hex 8 2>/dev/null || head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
LOG_FILE="import_logs/langroid_import_${unique_id}.log"

echo "Generating log file: ${LOG_FILE}"

# Directory creation
mkdir -p import_logs
echo "Created/verified 'import_logs' directory."

# Check for command-line argument (Python script path)
if [ -n "$1" ]; then
  PYTHON_SCRIPT="$1"
  echo "Profiling imports for script: ${PYTHON_SCRIPT}"
else
  # Default import statement (if no script provided)
  PYTHON_SCRIPT="-c 'import langroid'"
  echo "No script provided.  Profiling default 'import langroid'."
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

echo "Import profiling process completed."