# Handler Parameter Analysis Notes

## Overview

This document summarizes the investigation into how Langroid analyzes handler method parameters in `langroid/agent/base.py`, specifically focusing on the `_analyze_handler_params` method and its role in creating handler wrappers.

## Key Methods and Call Chain

### Call Chain
1. `_get_tool_list()` - Registers tool messages and their handlers
2. `_create_handler_wrapper()` - Creates wrapper functions for handlers
3. `_analyze_handler_params()` - Analyzes handler method signatures

## How _analyze_handler_params Works

The `_analyze_handler_params` method (lines 253-300 in agent/base.py) analyzes a handler method's signature to identify:
- Whether it has type annotations
- Which parameter is the agent parameter
- Which parameter is the chat_doc parameter

### Analysis Process
1. **Type Annotation Check**: First checks if parameters have type annotations
   - Looks for `Agent` in annotation string
   - Checks if annotation matches the agent's class
   - Looks for `ChatDocument` or `ChatDoc` in annotation string

2. **Fallback to Parameter Names**: If no annotations found
   - Looks for parameter named `agent`
   - Looks for parameter named `chat_doc`

## How _create_handler_wrapper Works

Based on the analysis from `_analyze_handler_params`, the wrapper creates different function signatures:
- No parameters → `wrapper(obj)`
- Both agent and chat_doc → `wrapper(obj, chat_doc)` with correct parameter order
- Only agent → `wrapper(obj)` passing agent internally
- Only chat_doc → `wrapper(obj, chat_doc)`

## Why Static Analysis Instead of Runtime

We investigated whether runtime type checking could be used instead of static analysis:

### Why Static Analysis is Necessary
1. **Setup Time Constraints**: Wrappers are created when tools are registered, before any actual calls
2. **Fixed Signatures**: Handler methods have specific expected signatures that must be matched
3. **Performance**: Analyzing once at setup avoids overhead on every call

### Why Runtime Analysis Isn't Feasible
1. Would require generic wrappers that accept `*args, **kwargs`
2. Would need to introspect handler signature on every call
3. Creates ambiguity about parameter ordering and names
4. Adds complexity without clear benefits

## Current Design Benefits
- Analyzes handler signatures once at setup time
- Creates wrappers with exact signatures needed
- No runtime ambiguity about parameter arrangement
- Clear error messages if handler signatures don't match expectations

## Related PR
This investigation was prompted by PR #861 "MCP updates" which made changes to how `FastMCPServer` forwards image context and resources, and added optional persistence for MCP server connections.