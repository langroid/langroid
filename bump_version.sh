#!/bin/sh
cz bump --increment $1
git commit pyproject.toml -m "Bump version"
cz version -p | cut -d' ' -f2
