# Quiet Mode - Suppressing Verbose Agent Output

Suppress Langroid's verbose agent output while showing your own custom progress.

## Key Imports

```python
from langroid.utils.configuration import quiet_mode, settings
```

## Context Manager (Recommended)

```python
from langroid.utils.configuration import quiet_mode

# Wrap agent runs in quiet_mode context
print("Starting writer...")

with quiet_mode():
    result = writer_task.run("Write the proposal")

print(f"Done! {len(result)} chars")
```

## Global Setting

```python
from langroid.utils.configuration import settings

settings.quiet = True   # Enable globally
result = task.run(...)
settings.quiet = False  # Disable
```

## What Gets Suppressed

- Agent streaming output
- Intermediate messages and tool outputs
- Rich console spinners/status messages
- Response statistics (show_stats)
- Debug information

## Pattern: Multi-Step Workflow with Progress

```python
from langroid.utils.configuration import quiet_mode

def run_workflow():
    print("Phase 1: Writing proposal...")
    with quiet_mode():
        proposal = writer_task.run("Write proposal")
    print(f"  ✓ Proposal written ({len(proposal)} chars)")

    print("Phase 2: Reviewing...")
    with quiet_mode():
        edits = reviewer_task.run(f"Review:\n{proposal}")
    print(f"  ✓ Found {len(edits)} issues")

    for i, edit in enumerate(edits, 1):
        print(f"  Applying edit {i}/{len(edits)}...")
        with quiet_mode():
            result = editor_task.run(edit)
        print(f"    ✓ Applied")

    print("Done!")
```

## Thread Safety

- Uses thread-local storage
- Supports nesting (once quiet, stays quiet in nested contexts)
- Exception-safe (reverts even on error)

```python
with quiet_mode():
    with quiet_mode(quiet=False):
        # Still quiet - once enabled, stays enabled in nesting
        assert settings.quiet
```

## Key Files in Langroid Repo

- `langroid/utils/configuration.py` - Main implementation (lines 111-128)
- `langroid/utils/output/status.py` - Status output helper
- `langroid/agent/batch.py` - Real-world usage example
- `tests/main/test_quiet_mode.py` - Test examples
