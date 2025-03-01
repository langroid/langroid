#!/bin/bash

echo "Starting import profiling without cache..."

# Ensure script argument or default to 'langroid'
if [ -n "$1" ]; then
  PYTHON_SCRIPT="$1"
  SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)  # Extract script name without extension
  echo "Profiling imports for script: ${PYTHON_SCRIPT}"
else
  SCRIPT_NAME="langroid"
  echo "import langroid" > /tmp/temp_import_script.py
  PYTHON_SCRIPT="/tmp/temp_import_script.py"
  echo "No script provided. Profiling default 'import langroid'."
fi

# Log file name creation with script reference
unique_id=$(openssl rand -hex 8 2>/dev/null || head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
LOG_FILE="import_logs/${SCRIPT_NAME}_import_${unique_id}.log"

echo "Generating log file: ${LOG_FILE}"

# Directory creation
mkdir -p import_logs
echo "Created/verified 'import_logs' directory."

# Run Python import profiling without cache
echo "Profiling imports using Python's -X importtime with no cache..."
PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0 python -I -B -X importtime "$PYTHON_SCRIPT" >> "${LOG_FILE}" 2>&1

echo "Import profiling completed."

# Run Tuna if installed
if python -c "import tuna" 2>/dev/null; then
  echo "Running tuna on ${LOG_FILE}..."
  tuna "${LOG_FILE}"
else
  echo "Error: The 'tuna' Python package is not installed."
  echo "Please install it using: pip install tuna"
  exit 1
fi

# Clean up temporary script if it was created
if [ "$SCRIPT_NAME" = "langroid" ]; then
  rm /tmp/temp_import_script.py
  echo "Removed temporary script file."
fi

echo "Import profiling process completed."
