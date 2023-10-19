.PHONY: setup check lint tests docs nodocs loc

SHELL := /bin/bash

.PHONY: setup update

setup: ## Setup the git pre-commit hooks
	poetry run pre-commit install

update: ## Update the git pre-commit hooks
	poetry run pre-commit autoupdate

check:
	@poetry run pre-commit install
	@poetry run pre-commit autoupdate
	@poetry run pre-commit run --all-files
	@echo "Running black..."
	@black --check .
	@echo "Running flake8 on git-tracked files ONLY! ..."
	@git ls-files | grep '\.py$$' | xargs flake8 --exclude=.git,__pycache__,.venv,__init__.py
	@poetry run ruff .
	@echo "Running mypy...";
	@poetry run mypy -p langroid
	@echo "All checks passed!"

lint:
	black .
	poetry run ruff . --fix

tests:
	pytest tests/main --basetemp=/tmp/pytest

docs:
	@# Kill any existing 'mkdocs serve' processes.
	@pkill -f "mkdocs serve" 2>/dev/null || true
	@# Build the documentation.
	mkdocs build
	@# Serve the documentation in the background.
	mkdocs serve &
	@echo "Documentation is being served in the background."
	@echo "You can access the documentation at http://127.0.0.1:8000/"

nodocs:
	@# Kill any existing 'mkdocs serve' processes.
	@pkill -f "mkdocs serve" 2>/dev/null || echo "No 'mkdocs serve' process found."
	@echo "Stopped serving documentation."


loc:
	@echo "Lines in git-tracked files python files:"
	@git ls-files | grep '\.py$$' | xargs cat | grep -v '^\s*$$' | wc -l

.PHONY: bump-patch
bump-patch:
	@poetry version patch
	@git commit pyproject.toml -m "Bump version"

.PHONY: bump-minor
bump-minor:
	@poetry version minor
	@git commit pyproject.toml -m "Bump version"

.PHONY: bump-major
bump-major:
	@poetry version major
	@git commit pyproject.toml -m "Bump version"

.PHONY: build
build:
	@poetry build

.PHONY: push
push:
	@git push origin main

.PHONY: clean
clean:
	-rm -rf dist/*

.PHONY: release
release:
	@VERSION=$$(poetry version | cut -d' ' -f2) && gh release create $${VERSION} dist/*

.PHONY: all-patch
all-patch: bump-patch clean build push release

.PHONY: all-minor
all-minor: bump-minor clean build push release

.PHONY: all-major
all-major: bump-major clean build push release

.PHONY: publish
publish:
	poetry publish
