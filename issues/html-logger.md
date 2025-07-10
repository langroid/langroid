# HTML Logger Specification for Langroid Task System

## Overview

This document specifies the requirements for a new HTML logger that will enhance
the current logging system in Langroid's task.py module. The HTML logger will
produce self-contained HTML files with collapsible entries, providing a more
user-friendly way to navigate complex multi-agent conversations.

## Current State

The Langroid task system currently supports two logging formats:
1. **TSV Logger**: Tab-separated values for structured data analysis
2. **Plain Text Logger**: Rich-formatted text logs with color coding

Both loggers capture comprehensive information about agent interactions, including
task names, responders, message content, and tool usage.

## Requirements

### 1. Output Format

- **File Type**: Self-contained HTML file with embedded CSS and JavaScript
- **File Extension**: `.html`
- **File Naming**: Same pattern as existing loggers: `{task_name}.html`
- **Encoding**: UTF-8

### 2. Visual Structure

#### 2.1 Overall Layout
- Dark theme with dark gray/black background (#2b2b2b or similar)
- Monospace font for consistency with terminal output
- Fixed header showing model info and timestamp
- Responsive design that works on various screen sizes
- Golden/amber accent color for headers and borders (#d4a017 or similar)

#### 2.2 Fixed Header Section
- Model name and version (e.g., "claude-opus-4-20250514")
- Timestamp of log generation
- Total message count
- Styled with golden border and text

#### 2.3 Collapsible Entries
Each log entry must be collapsible with:
- **Collapsed State**: Shows only the entity type/role
- **Expanded State**: Shows full message content with sub-sections
- **Toggle Control**: [+] and [-] text indicators in square brackets

#### 2.4 Entry Structure
Each entry consists of:
- **Role Header**: Entity type in colored uppercase (USER, ASSISTANT, SYSTEM, etc.)
- **Collapsible Sections**: Each with [+]/[-] toggle:
  - System Prompt (if applicable)
  - Tools (with count)
  - System Reminder (if applicable)
  - Main content

#### 2.5 Color Scheme
- **USER**: Green text (#00ff00 or similar)
- **ASSISTANT**: Red/orange text (#ff6b6b or similar)
- **SYSTEM**: Gray text
- **Tool calls**: Green indicators for [+] toggles
- **Tool results**: Success (✓) in green, Error (✗) in red
- **Code blocks**: Dark background with syntax highlighting

#### 2.6 Tool Display
When expanded, tool calls should show:
- Tool name and parameters in a code block
- Tool result with success/error indicator
- Raw tool call details (collapsible sub-section)

Example structure:
```
ASSISTANT
[+] System Reminder

I'll read the langroid-llms.txt file to see what it contains.

  [+] Read(/Users/pchalasani/Git/claude-code-play/langroid-llms.txt)
  [+] Tool Result ✓
  [-] Raw Tool Call
      {
        "type": "tool_use",
        "id": "toolu_0184van1ug4T6kAj7a8SkaKp",
        "name": "Read",
        "input": {
          "file_path": "/Users/pchalasani/Git/claude-code-play/langroid-llms.txt"
        }
      }
```

### 3. Functionality

#### 3.1 User Controls
- **Expand/Collapse Individual**: Click on entry header or toggle button
- **Expand All**: Button to expand all entries
- **Collapse All**: Button to collapse all entries
- **Search**: Basic text search functionality (optional enhancement)

#### 3.2 State Persistence
- Collapse/expand state should be maintained during the session
- No requirement for persistence across page reloads

### 4. Data Representation

The HTML logger should capture all information currently logged by the plain
text logger and organize it hierarchically:

#### 4.1 Primary Level (Always Visible)
- Entity/Role name (USER, ASSISTANT, AGENT, etc.)
- Task name prefix if not "root"

#### 4.2 Collapsible Sections
Each entry may have multiple collapsible sub-sections:
- **System Messages**: System prompts, reminders, etc.
- **Tool Information**: 
  - Tool count in header (e.g., "Tools (17)")
  - Individual tool calls with name and parameters
  - Tool results with success/error indicators
  - Raw tool call JSON (nested collapsible)
- **Message Content**: The actual text content
- **Metadata** (when relevant):
  - Recipient information
  - Blocked entities
  - Mark indicator for final results

#### 4.3 Mapping from Current Log Fields
- `responder` → Entity type (USER, ASSISTANT, etc.)
- `task_name` → Prefix before entity if not "root"
- `sender` + `sender_name` → Combined in display
- `tool_type` + `tool` → Tool section with appropriate formatting
- `content` → Main message content
- `mark` → Special indicator for final results
- `recipient` → Shown in metadata when present
- `block` → Shown in metadata when present

### 5. Integration Requirements

#### 5.1 Implementation Location
- Add to the existing `init_loggers` method in task.py
- Follow the same pattern as TSV and plain text loggers
- Use the same log directory and naming conventions

#### 5.2 Configuration
- HTML logger should be optional
- Controlled via configuration flag or environment variable
- Should not interfere with existing loggers

#### 5.3 Compatibility
- Must work with the existing `log_message` method
- Support the same ChatDocLoggerFields structure
- Handle sub-tasks transparently (no special handling needed)

### 6. Performance Considerations

- Efficient for files with thousands of log entries
- Minimal JavaScript for toggle functionality
- CSS animations should be optional or lightweight
- File size should remain reasonable for large conversations

### 7. Accessibility

- Keyboard navigation support for expanding/collapsing entries
- Clear visual indicators for interactive elements
- Sufficient color contrast for readability
- Screen reader compatible structure

## Example Visual Mock-up

```
claude-opus-4-20250514
7/8/2025, 12:00:50 PM   8 messages
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[+] System Prompt
[+] Tools (17)

USER
[+] System Reminder

Use the read tool to read the langroid-llms.txt and see what it is about

[+] System Reminder

ASSISTANT

I'll read the langroid-llms.txt file to see what it contains.

  [+] Read(/Users/pchalasani/Git/claude-code-play/langroid-llms.txt)
  [+] Tool Result ✗
  [-] Raw Tool Call
      {
        "type": "tool_use",
        "id": "toolu_0184van1ug4T6kAj7a8SkaKp",
        "name": "Read",
        "input": {
          "file_path": "/Users/pchalasani/Git/claude-code-play/langroid-llms.txt"
        }
      }

ASSISTANT

The file is quite large (3.3MB). Let me read it in smaller chunks to understand its content.

  [-] Read(/Users/pchalasani/Git/claude-code-play/langroid-llms.txt)
      {
        "file_path": "/Users/pchalasani/Git/claude-code-play/langroid-llms.txt",
        "limit": 100
      }
  [+] Tool Result ✓
  [+] Raw Tool Call
```

## Future Enhancements (Out of Scope)

These features are not required for the initial implementation but could be
added later:
- Filtering by entity type, task name, or tool
- Export to other formats
- Real-time log streaming
- Syntax highlighting for code in messages
- Timestamp display options
- Log entry grouping by conversation threads