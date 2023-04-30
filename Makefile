.PHONY: check

check:
	@echo "Running black..."
	@black --check .
	@echo "Running flake8 on git-tracked files ONLY! ..."
	@git ls-files | grep '\.py$$' | xargs flake8 --exclude=.git,__pycache__,.venv
	@echo "Running mypy..."
	@mypy -p llmagent
	@echo "All checks passed!"
