.PHONY: setup check lint tests docs nodocs loc

## set branch-suffix to current branch name, unless it is main (in which case it is empty)
## This allows publishing "special" versions from branches other than main,
## e.g. 0.6.5neo4j
BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)
SHELL := /bin/bash

.PHONY: setup update

setup: ## Setup the git pre-commit hooks
	poetry run pre-commit install

update: ## Update the git pre-commit hooks
	poetry run pre-commit autoupdate

.PHONY: type-check
type-check:
	@poetry run pre-commit install
	@poetry run pre-commit autoupdate
	@poetry run pre-commit run --all-files
	@echo "Running black..."
	@black --check .
	@echo "Running flake8 on git-tracked files ONLY! ..."
	@git ls-files | grep '\.py$$' | xargs flake8 --exclude=.git,__pycache__,.venv,langroid/embedding_models/protoc/*
	@poetry run ruff .
	@echo "Running mypy...";
	@poetry run mypy -p langroid
	@echo "All checks passed!"

.PHONE: lint
lint:
	black .
	poetry run ruff . --fix

.PHONY: stubs
stubs:
	@echo "Generating Python stubs for the langroid package..."
	@poetry run stubgen -p langroid -o stubs
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


.PHONY: test-condition
test-condition:
	@if [ "$(BRANCH_NAME)" = "main" ]; then \
		echo "On main branch"; \
	else \
		echo "On branch $(BRANCH_NAME)"; \
	fi

.PHONY: bump-patch
bump-patch:
	@echo "Branch Name: $(BRANCH_NAME)"
	@if [ "$(BRANCH_NAME)" = "main" ]; then \
		echo "Bumping version for main branch"; \
		bumpversion patch; \
	else \
		echo "Bumping version for $(BRANCH_NAME)"; \
		NEW_VERSION=$$(bumpversion --dry-run --list patch | grep new_version | sed -r "s/new_version=//")-$(BRANCH_NAME); \
		echo "New Version: $$NEW_VERSION"; \
		bumpversion patch --new-version $$NEW_VERSION; \
	fi

.PHONY: bump-minor
bump-minor:
	@echo "Branch Name: $(BRANCH_NAME)"
	@if [ "$(BRANCH_NAME)" = "main" ]; then \
		echo "Bumping version for main branch"; \
		bumpversion minor; \
	else \
		echo "Bumping version for $(BRANCH_NAME)"; \
		NEW_VERSION=$$(bumpversion --dry-run --list minor | grep new_version | sed -r "s/new_version=//")-$(BRANCH_NAME); \
		echo "New Version: $$NEW_VERSION"; \
		bumpversion minor --new-version $$NEW_VERSION; \
	fi

.PHONY: bump-major
bump-major:
	@echo "Branch Name: $(BRANCH_NAME)"
	@if [ "$(BRANCH_NAME)" = "main" ]; then \
		echo "Bumping version for main branch"; \
		bumpversion major; \
	else \
		echo "Bumping version for $(BRANCH_NAME)"; \
		NEW_VERSION=$$(bumpversion --dry-run --list major | grep new_version | sed -r "s/new_version=//")-$(BRANCH_NAME); \
		echo "New Version: $$NEW_VERSION"; \
		bumpversion major --new-version $$NEW_VERSION; \
	fi

.PHONY: build
build:
	@poetry build

.PHONY: push
push:
	@git push origin $(shell git rev-parse --abbrev-ref HEAD)

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
