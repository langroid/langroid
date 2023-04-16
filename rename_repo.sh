#!/bin/bash

# Function to prompt for user confirmation
confirm() {
  read -p "$1 (y/n): " choice
  case "$choice" in
    [Yy]* ) return 0;;
    [Nn]* ) return 1;;
    * ) echo "Please answer y (yes) or n (no)."; return 1;;
  esac
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <new-name>"
  exit 1
fi

# Assign command line argument to variable
new_name="$1"

# Extract the GitHub username and old name from the current repository's remote URL
remote_url=$(git remote get-url origin)
github_username=$(echo "$remote_url" | sed -n 's/.*github.com[:/]\([^/]*\)\/.*/\1/p')
old_name=$(echo "$remote_url" | sed -n 's/.*\/\([^/]*\)\.git/\1/p')

# Ask for confirmation and rename directories containing the old name
cmd="find . -type d -name '*$old_name*' -execdir bash -c 'mv \"\$0\" \"\${0/$old_name/$new_name}\"' {} \;"
if confirm "Execute: $cmd"; then
  eval "$cmd"
fi

# Ask for confirmation and rename files containing the old name
cmd="find . -type f -name '*$old_name*' -execdir bash -c 'mv \"\$0\" \"\${0/$old_name/$new_name}\"' {} \;"
if confirm "Execute: $cmd"; then
  eval "$cmd"
fi

# Ask for confirmation and replace strings within files containing the old name
# Ask for confirmation and replace strings within files containing the old name
if confirm "Replace strings within files containing the old name?"; then
  find . -type f -name "*$old_name*" -exec sh -c 'sed -i.bak "s/'"$old_name"'/'"$new_name"'/g" "$1" && rm "$1.bak"' sh {} \;
fi

# Ask for confirmation and update the remote URL of the repository to reflect the new name
cmd="git remote set-url origin 'https://github.com/$github_username/$new_name.git'"
if confirm "Execute: $cmd"; then
  eval "$cmd"
  echo "Renaming complete."
else
  echo "Could not update the remote URL of the repository."
fi
