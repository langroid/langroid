# Langroid v0.56.4 Release Notes

## Improved Handler Parameter Analysis for Tool Messages

### Overview
This release enhances the internal mechanism for analyzing handler method parameters in `ToolMessage` handlers, providing more robust and accurate type detection.

### Key Improvements

#### Direct Type Checking for Handler Parameters
- **Agent parameter detection**: Now uses direct class checking with `inspect.isclass()` and `issubclass()` for more accurate detection of Agent-typed parameters
- **ChatDocument detection**: Uses direct identity comparison (`param.annotation is ChatDocument`) for exact type matching
- **Complex type support**: Maintains fallback to string-based detection for complex generic types like `Optional[Agent]`

#### Better Parameter Extraction
- Improved the method for removing the `self` parameter from handler signatures using index slicing instead of name-based filtering
- More reliable parameter analysis for both synchronous and asynchronous handlers

### Why This Matters
These improvements make handler parameter detection more robust, especially when working with:
- Subclasses of `Agent` 
- Tools that require specific agent or chat document context
- MCP (Model Context Protocol) tool handlers that use various parameter combinations

### Backward Compatibility
All existing handler patterns continue to work as before. The improvements are internal optimizations that enhance reliability without changing the API.

### Developer Impact
No code changes required. Handlers with type annotations like:
```python
def handle(self, agent: Agent, chat_doc: ChatDocument) -> str:
    ...
```
will benefit from more accurate parameter detection and routing.

### Related Changes
- Removed debug print statement from `_analyze_handler_params` method
- Enhanced test coverage for MCP tools with various handler signatures