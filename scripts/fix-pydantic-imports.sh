#!/bin/bash

# Define the directories to search in
directories=("langroid" "examples" "tests")

# Function to perform replacements and log changes
replace_and_log() {
    # Use find to locate all .py files in the specified directories
    find "${directories[@]}" -type f -name '*.py' | while read -r file; do
        # Check and replace lines starting with specific patterns
        if grep -q '^from pydantic ' "$file"; then
            sed -i'' -e  's/^from pydantic /from langroid.pydantic_v1 /' "$file"
            echo "Replaced 'from pydantic ' in $file"
        fi
        if grep -q '^from pydantic.v1 ' "$file"; then
            sed -i'' -e 's/^from pydantic.v1 /from langroid.pydantic_v1 /' "$file"
            echo "Replaced 'from pydantic.v1 ' in $file"
        fi
        if grep -q '^import pydantic' "$file"; then
            sed -i'' -e 's/^import pydantic/import langroid.pydantic_v1/' "$file"
            echo "Replaced 'import pydantic' in $file"
        fi
    done
}

# Call the function to perform the replacements and logging
replace_and_log
