# v0.56.14 - Response Sequence Tracking for DoneSequence

## Overview
Improved DoneSequence implementation by introducing response sequence tracking at the Task level, replacing the previous approach that relied on parent pointer traversal or agent message history.

## Changes

### Task Response Sequence Tracking
- Added `response_sequence: List[ChatDocument]` to track messages as the task executes
- Messages are added to the sequence after each `step()` in the `run()` method
- Duplicate messages are prevented by checking if the pending message ID differs from the last element

### Simplified Message Chain Retrieval
- `_get_message_chain()` now simply returns the last N elements from `response_sequence`
- Eliminates complexity of parent pointer traversal and agent boundary issues
- More efficient and reliable message chain tracking

## Benefits
- Better encapsulation: Task maintains its own response sequence
- More explicit control over what gets added to the sequence
- Cleaner implementation without reaching into agent internals
- Fixes issues with DoneSequence incorrectly including messages from subtask agents

## Testing
All existing done sequence tests pass without modification, confirming backward compatibility.