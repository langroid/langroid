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
	@echo "Running mypy...";
	@uv run mypy -p langroid
	@echo "All checks passed!"

.PHONY: lint
lint:
	uv run black .
	uv run ruff check . --fix
	@echo "Auto-fixing issues in examples folder..."
	@uv run ruff check examples/ --fix-only --no-force-exclude

.PHONY: stubs
stubs:
	@echo "Generating Python stubs for the langroid package..."
	@uv run stubgen -p langroid -o stubs
	@echo "Stubs generated in the 'stubs' directory"

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

.PHONY: repomix repomix-no-tests repomix-all

repomix: ## Generate llms.txt and llms-compressed.txt (includes tests)
	@echo "Generating llms.txt (with tests)..."
	@git ls-files | repomix --stdin
	@echo "Generating llms-compressed.txt..."
	@git ls-files | repomix --stdin --compress -o llms-compressed.txt
	@echo "Generated llms.txt and llms-compressed.txt"

repomix-no-tests: ## Generate llms-no-tests.txt (excludes tests)
	@echo "Generating llms-no-tests.txt (without tests)..."
	@git ls-files | grep -v "^tests/" | repomix --stdin -o llms-no-tests.txt
	@echo "Generating llms-no-tests-compressed.txt..."
	@git ls-files | grep -v "^tests/" | repomix --stdin --compress -o llms-no-tests-compressed.txt
	@echo "Generated llms-no-tests.txt and llms-no-tests-compressed.txt"

repomix-no-tests-no-examples: ## Generate llms-no-tests-no-examples.txt (excludes tests and examples)
	@echo "Generating llms-no-tests-no-examples.txt (without tests and examples)..."
	@git ls-files | grep -v -E "^(tests|examples)/" | repomix --stdin -o llms-no-tests-no-examples.txt
	@echo "Generating llms-no-tests-no-examples-compressed.txt..."
	@git ls-files | grep -v -E "^(tests|examples)/" | repomix --stdin --compress -o llms-no-tests-no-examples-compressed.txt
	@echo "Generated llms-no-tests-no-examples.txt and llms-no-tests-no-examples-compressed.txt"

repomix-all: repomix repomix-no-tests repomix-no-tests-no-examples ## Generate all repomix variants

.PHONY: check
check: lint type-check repomix-all

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

.PHONY: bump-rc
bump-rc:
	@cz bump --prerelease rc

.PHONY: bump-beta
bump-beta:
	@cz bump --prerelease beta

.PHONY: bump-alpha
bump-alpha:
	@cz bump --prerelease alpha

.PHONY: all-patch
all-patch: bump-patch clean build push release

.PHONY: all-minor
all-minor: bump-minor clean build push release

.PHONY: all-major
all-major: bump-major clean build push release

.PHONY: all-rc
all-rc: bump-rc clean build push release

.PHONY: all-beta
all-beta: bump-beta clean build push release

.PHONY: all-alpha
all-alpha: bump-alpha clean build push release

.PHONY: pre-release-branch
pre-release-branch: ## Create and push pre-release from current branch
	@CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) && \
	if [ "$$CURRENT_BRANCH" = "main" ]; then \
		echo "Error: Cannot create pre-release from main branch"; \
		exit 1; \
	fi && \
	PRERELEASE_TYPE=$${PRERELEASE_TYPE:-rc} && \
	cz bump --prerelease "$$PRERELEASE_TYPE" && \
	VERSION=$$(cz version -p | cut -d' ' -f2) && \
	echo "Creating pre-release $$VERSION from branch $$CURRENT_BRANCH" && \
	git push origin "$$CURRENT_BRANCH" --tags && \
	gh release create "$$VERSION" dist/* --target "$$CURRENT_BRANCH" --prerelease --title "Pre-release $$VERSION" --notes "Experimental pre-release from $$CURRENT_BRANCH"

.PHONY: pre-release-rc
pre-release-rc: ## Create release candidate from current branch
	@PRERELEASE_TYPE=rc make pre-release-branch

.PHONY: pre-release-beta
pre-release-beta: ## Create beta release from current branch
	@PRERELEASE_TYPE=beta make pre-release-branch

.PHONY: pre-release-alpha
pre-release-alpha: ## Create alpha release from current branch
	@PRERELEASE_TYPE=alpha make pre-release-branch

.PHONY: pre-release-push
pre-release-push: ## Push current branch and tags (for pre-releases)
	@CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) && \
	if [ "$$CURRENT_BRANCH" = "main" ]; then \
		echo "Error: Cannot push pre-release from main branch"; \
		exit 1; \
	fi && \
	git push origin "$$CURRENT_BRANCH" --tags

.PHONY: pre-release-release
pre-release-release: ## Create GitHub pre-release (requires VERSION env var)
	@CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) && \
	if [ "$$CURRENT_BRANCH" = "main" ]; then \
		echo "Error: Cannot create pre-release from main branch"; \
		exit 1; \
	fi && \
	VERSION=$$(cz version -p | cut -d' ' -f2) && \
	echo "Creating pre-release $$VERSION from branch $$CURRENT_BRANCH" && \
	gh release create "$$VERSION" dist/* --target "$$CURRENT_BRANCH" --prerelease --title "Pre-release $$VERSION" --notes "Experimental pre-release from $$CURRENT_BRANCH"

.PHONY: bump-rc-patch
bump-rc-patch: ## Bump to release candidate patch
	@cz bump --increment PATCH --prerelease rc

.PHONY: bump-rc-minor
bump-rc-minor: ## Bump to release candidate minor
	@cz bump --increment MINOR --prerelease rc

.PHONY: bump-rc-major
bump-rc-major: ## Bump to release candidate major
	@cz bump --increment MAJOR --prerelease rc

.PHONY: bump-beta-patch
bump-beta-patch: ## Bump to beta patch
	@cz bump --increment PATCH --prerelease beta

.PHONY: bump-beta-minor
bump-beta-minor: ## Bump to beta minor
	@cz bump --increment MINOR --prerelease beta

.PHONY: bump-alpha-patch
bump-alpha-patch: ## Bump to alpha patch
	@cz bump --increment PATCH --prerelease alpha

.PHONY: bump-alpha-minor
bump-alpha-minor: ## Bump to alpha minor
	@cz bump --increment MINOR --prerelease alpha

.PHONY: pre-release-rc-patch
pre-release-rc-patch: bump-rc-patch clean build pre-release-push pre-release-release

.PHONY: pre-release-rc-minor
pre-release-rc-minor: bump-rc-minor clean build pre-release-push pre-release-release

.PHONY: pre-release-rc-major
pre-release-rc-major: bump-rc-major clean build pre-release-push pre-release-release

.PHONY: pre-release-beta-patch
pre-release-beta-patch: bump-beta-patch clean build pre-release-push pre-release-release

.PHONY: pre-release-beta-minor
pre-release-beta-minor: bump-beta-minor clean build pre-release-push pre-release-release

.PHONY: pre-release-alpha-patch
pre-release-alpha-patch: bump-alpha-patch clean build pre-release-push pre-release-release

.PHONY: pre-release-alpha-minor
pre-release-alpha-minor: bump-alpha-minor clean build pre-release-push pre-release-release

.PHONY: publish
publish:
	uv publish
