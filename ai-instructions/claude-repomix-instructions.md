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

### 5. Generate Two Output Versions
```bash
# Standard version (uncompressed)
repomix

# Compressed version (smaller file size)
repomix --compress -o llms-compressed.txt
```

### 6. Verify Results
- Check file sizes and token counts in repomix output
- Ensure no sensitive data is included
- Confirm only relevant source files are packaged

## Expected Outcome
Two text files optimized for different LLM context windows:
- `llms.txt`: Full version for detailed analysis
- `llms-compressed.txt`: Compressed version for general use

The files should contain only git-tracked source code with proper exclusions for clean, focused LLM consumption.