#!/bin/bash

# Langroid currently has pydantic v2 compatibility, but internally uses v1,
# via langroid.pydantic_v1. However since `import pydantic` brings in v2,
# this script replaces all instances of 'from pydantic' and 'import pydantic' in
# Python files with 'from langroid.pydantic_v1' and 'import langroid.pydantic_v1'.
#
# It makes an exception if the line contains '# keep', and leaves the
# import untouched. Of course this should be used mainly in tests and examples,
# since we don't want to mix pydantic v1 and v2 within core langroid code.

# Define the directories to search in
directories=("langroid" "examples" "tests")

# Function to perform replacements and log changes
replace_and_log() {
    # Use find to locate all .py files in the specified directories, excluding .venv directories
    find "${directories[@]}" -type f -name '*.py' -not -path '*/.venv/*' | while read -r file; do
        # Check and replace lines starting with specific patterns
        if grep -q '^from pydantic ' "$file" && grep -v '# keep' "$file" | grep -q '^from pydantic '; then
            sed -i'' -e  '/^from pydantic .*# keep/!s/^from pydantic /from langroid.pydantic_v1 /' "$file"
            echo "Replaced 'from pydantic ' in $file"
        fi
        if grep -q '^from pydantic.v1 ' "$file" && grep -v '# keep' "$file" | grep -q '^from pydantic.v1 '; then
            sed -i'' -e '/^from pydantic .*# keep/s/^from pydantic.v1 /from langroid.pydantic_v1 /' "$file"
            echo "Replaced 'from pydantic.v1 ' in $file"
        fi
        if grep -q '^import pydantic' "$file" && grep -v '# keep' "$file" | grep -q '^import pydantic'; then
            sed -i'' -e '/^from pydantic .*# keep/!s/^import pydantic/import langroid.pydantic_v1/' "$file"
            echo "Replaced 'import pydantic' in $file"
        fi
    done
}

# Call the function to perform the replacements and logging
replace_and_log
