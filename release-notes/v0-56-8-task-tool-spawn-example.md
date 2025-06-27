# v0.56.8 Release Notes

## 🚀 New Features

### TaskTool Dynamic Sub-Agent Spawning Example

- Added `examples/basic/planner-workflow-spawn.py` demonstrating how to use `TaskTool` to dynamically spawn specialized sub-agents during execution
- Example shows a planner agent that solves multi-step math problems by spawning incremental and doubling agents as needed
- Showcases the power of dynamic agent creation without pre-defining sub-agents in the main script

## 🧪 Testing

- Added comprehensive tests for `TaskTool` including support for `tools="ALL"` option
- Enhanced test coverage for dynamic sub-agent spawning scenarios

## 🛠️ Development Improvements

### Ruff Auto-Fix for Examples

- Updated Makefile to run `ruff check examples/ --fix-only` to automatically fix code style issues in examples
- Removed F401 (unused imports) from ruff's ignore list to catch and fix unused imports
- Auto-fixed imports in 150+ example files for better code consistency
- Examples folder remains excluded from error reporting but benefits from automatic fixes

## 🔧 Configuration Changes

- Commented out flake8 in favor of ruff for linting (ruff is faster and covers all flake8 rules)
- Updated `pyproject.toml` to enable F401 checking
- Modified Makefile to add `--no-force-exclude` flag for ruff when processing examples