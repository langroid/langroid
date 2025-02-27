#!/bin/bash

echo "Starting import profiling..."

# Log file name creation
unique_id=$(openssl rand -hex 8 2>/dev/null || head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
LOG_FILE="import_logs/langroid_import_${unique_id}.log"

echo "Generating log file: ${LOG_FILE}"

# Directory creation
mkdir -p import_logs
echo "Created/verified 'import_logs' directory."

# Langroid import (with logging)
echo "Profiling imports using Python's -X importtime..."
python -X importtime -c "import langroid" >> "${LOG_FILE}" 2>&1
echo "Langroid import profiling completed."

# If you have the script you want to test just add it here like
# and uncomment this line
# python -X importtime <path_to_your_script.py>  >> "${LOG_FILE}" 2>&1

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

