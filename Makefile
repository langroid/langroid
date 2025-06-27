.PHONY: setup check lint tests docs nodocs loc

SHELL := /bin/bash

.PHONY: setup update

setup: ## Setup the git pre-commit hooks
	uv run pre-commit install

update: ## Update the git pre-commit hooks
	uv run pre-commit autoupdate

.PHONY: type-check
type-check:
	@uv run pre-commit install
	@uv run pre-commit autoupdate
	@uv run pre-commit run --all-files
	@echo "Running black..."
	@uv run black --check .
	@echo "Running ruff check (without fix)..."
	@uv run ruff check .
	@echo "Running flake8 on git-tracked files ONLY! ..."
	@git ls-files | grep '\.py$$' | xargs flake8 --exclude=.git,__pycache__,.venv,langroid/embedding_models/protoc/*
	@echo "Running mypy...";
	@uv run mypy -p langroid
	@echo "All checks passed!"

.PHONY: lint
lint:
	uv run black .
	uv run ruff check . --fix
	@echo "Auto-fixing issues in examples folder..."
	@uv run ruff check examples/ --fix-only

.PHONY: stubs
stubs:
	@echo "Generating Python stubs for the langroid package..."
	@uv run stubgen -p langroid -o stubs
	@echo "Stubs generated in the 'stubs' directory"

.PHONY: fix-pydantic

# Entry to replace pydantic imports in specified directories
fix-pydantic:
	@echo "Fixing pydantic imports..."
	@chmod +x scripts/fix-pydantic-imports.sh
	@./scripts/fix-pydantic-imports.sh

.PHONY: check
check: fix-pydantic lint type-check

.PHONY: tests
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

.PHONY: revert-tag
revert-tag:
	@LATEST_TAG=$$(git describe --tags --abbrev=0) && \
	echo "Deleting tag: $$LATEST_TAG" && \
	git tag -d $$LATEST_TAG

.PHONY: revert-bump
revert-bump:
	@if git log -1 --pretty=%B | grep -q "bump"; then \
		git reset --hard HEAD~1; \
		echo "Reverted last commit (bump commit)"; \
	else \
		echo "Last commit was not a bump commit"; \
	fi

.PHONY: revert
revert: revert-bump revert-tag
	
.PHONY: bump-patch
bump-patch:
	@cz bump --increment PATCH

.PHONY: bump-minor
bump-minor:
	@cz bump --increment MINOR 

.PHONY: bump-major
bump-major:
	@cz bump --increment MAJOR 

.PHONY: build
build:
	@uv build

.PHONY: push
push:
	@git push origin main
	@git push origin --tags

.PHONY: clean
clean:
	-rm -rf dist/*

.PHONY: release
release:
	@VERSION=$$(cz version -p | cut -d' ' -f2) && gh release create $${VERSION} dist/*

.PHONY: all-patch
all-patch: bump-patch clean build push release

.PHONY: all-minor
all-minor: bump-minor clean build push release

.PHONY: all-major
all-major: bump-major clean build push release

.PHONY: publish
publish:
	uv publish
