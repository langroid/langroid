# AI Instructions for Setting Up Repomix

## Task Overview
Set up [repomix](https://github.com/yamadashy/repomix) to generate LLM-friendly repository exports. This creates text files that can be uploaded to AI models for code analysis.

## Steps to Complete

### 1. Install Repomix
```bash
npm install -g repomix
```

### 2. Create repomix.config.json
Create a configuration file in the repository root with:
- **Include patterns**: Source code files (*.py, *.js, *.md, *.yaml, *.yml, *.toml)
- **Exclude patterns**: Data directories, logs, node_modules, JSON files, generated files
- **Security check**: Enable to prevent sensitive data inclusion

### 3. Configure Include/Exclude Patterns
- Include only source code directories and documentation
- Exclude data/, logs/, build artifacts, dependencies
- Add `llms*.txt` to exclusions to prevent recursive inclusion

### 4. Test Configuration (Optional)
```bash
# Generate file list only for inspection
repomix --no-files -o file-list.txt
```
This allows you to review which files will be included before generating the full output.

### 5. Generate Output Versions

Use the Makefile targets to generate repomix files:

```bash
# Generate all variants (recommended)
make repomix-all

# Or generate specific versions:
make repomix                      # llms.txt and llms-compressed.txt (includes tests)
make repomix-no-tests             # llms-no-tests.txt and llms-no-tests-compressed.txt
make repomix-no-tests-no-examples # llms-no-tests-no-examples.txt and compressed version
```

All commands use `git ls-files` to ensure only git-tracked files are included.

### 6. Verify Results
- Check file sizes and token counts in repomix output
- Ensure no sensitive data is included
- Confirm only relevant source files are packaged

## Expected Outcome
Six text files optimized for different LLM contexts:
- `llms.txt`: Full version with tests and examples (870K tokens)
- `llms-compressed.txt`: Compressed version with tests and examples (513K tokens)
- `llms-no-tests.txt`: Full version without tests (677K tokens)
- `llms-no-tests-compressed.txt`: Compressed version without tests (433K tokens)
- `llms-no-tests-no-examples.txt`: Core library code only (no tests/examples)
- `llms-no-tests-no-examples-compressed.txt`: Compressed core library code (285K tokens)

The files contain only git-tracked source code with proper exclusions for clean, focused LLM consumption.