# Release Notes - v0.56.2

## TableChatAgent Enhancement: Data Cleaning Support with `df.assign()`

### Overview
This release enhances the TableChatAgent to better support data cleaning operations while maintaining security. Users can now perform column transformations using the safe `df.assign()` method.

### Key Changes

#### 1. Enabled `df.assign()` Method
- Added `assign` to the whitelist of allowed DataFrame methods
- Provides a secure way to create modified DataFrames without allowing arbitrary assignments
- Maintains the existing security model while enabling common data cleaning tasks

#### 2. Improved Agent Guidance
- Updated system message to proactively explain that assignment statements (`df['col'] = ...`) are not allowed
- Clear guidance to use `df.assign()` for data modifications
- Agent now correctly uses `df.assign()` on first attempt, avoiding error-correction cycles

### Example Usage
When asked to clean data, the agent will now use:
```python
df.assign(airline=df['airline'].str.replace('*', ''))
```
Instead of attempting:
```python
df['airline'] = df['airline'].str.replace('*', '')  # This would fail
```

### Security Considerations
- The `assign` method is safe as it returns a new DataFrame without side effects
- Cannot be used for arbitrary code execution, file I/O, or network access
- Expressions passed to `assign` still go through the same sanitization process
- Maintains the eval-only security model (no exec)

### Testing
- Added comprehensive test coverage for self-correction behavior
- Verified agent successfully handles data cleaning requests

This addresses issue #867 and improves the TableChatAgent's utility for data cleaning workflows.