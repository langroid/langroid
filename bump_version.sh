#!/bin/sh
poetry version $1
git commit pyproject.toml -m "Bump version"
poetry version | cut -d' ' -f2
