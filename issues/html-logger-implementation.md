# HTML Logger Implementation Plan

## Overview

This document outlines the technical implementation plan for adding an HTML logger
to Langroid's task system. The implementation will create self-contained HTML files
with collapsible log entries, following the specification in `html-logger.md`.

## Architecture

### 1. Core Components

#### 1.1 HTMLLogger Class
Create a new logger class that inherits from or follows the pattern of existing loggers:

```python
class HTMLLogger:
    def __init__(self, filename: str, log_dir: str = "logs"):
        self.file_path = Path(log_dir) / f"{filename}.html"
        self.entries = []
        self._write_header()
    
    def log(self, fields: ChatDocLoggerFields):
        """Add a log entry"""
        entry = self._format_entry(fields)
        self.entries.append(entry)
        self._append_to_file(entry)
    
    def close(self):
        """Finalize the HTML file"""
        self._write_footer()
```

#### 1.2 HTML Template Structure
The HTML file will have three main sections:

1. **Header Section**: Static CSS, JavaScript, and page header
2. **Content Section**: Dynamic log entries
3. **Footer Section**: Closing tags and finalization

### 2. Implementation Steps

#### Step 1: Create HTML Logger Foundation
1. Add `html_logger.py` in `langroid/agent/logging/`
2. Define the `HTMLLogger` class with basic file handling
3. Implement HTML header generation with embedded CSS and JavaScript

#### Step 2: Integrate with Task System
1. Modify `init_loggers` method in `task.py` to include HTML logger option
2. Add configuration flag (e.g., `enable_html_logging` in TaskConfig)
3. Update `log_message` method to send data to HTML logger

#### Step 3: Implement HTML Generation
1. Create entry formatting logic that converts ChatDocLoggerFields to HTML
2. Implement hierarchical structure for collapsible sections
3. Add proper escaping for HTML special characters

#### Step 4: Add JavaScript Functionality
1. Implement toggle functionality for collapsible sections
2. Add "Expand All" / "Collapse All" controls
3. Ensure smooth animations and state management

### 3. Detailed Component Design

#### 3.1 HTML Header Template
```python
HTML_HEADER = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{task_name} - Langroid Task Log</title>
    <style>
        body {{
            background-color: #2b2b2b;
            color: #f0f0f0;
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            border: 2px solid #d4a017;
            padding: 10px;
            margin-bottom: 20px;
            color: #d4a017;
        }}
        .entry {{
            margin-bottom: 10px;
            border-left: 3px solid transparent;
        }}
        .entry.user {{ border-left-color: #00ff00; }}
        .entry.assistant {{ border-left-color: #ff6b6b; }}
        .toggle {{
            cursor: pointer;
            user-select: none;
            color: #00ff00;
        }}
        .collapsed .content {{ display: none; }}
        /* More styles... */
    </style>
    <script>
        function toggle(id) {{
            const element = document.getElementById(id);
            element.classList.toggle('collapsed');
            const toggle = element.querySelector('.toggle');
            toggle.textContent = element.classList.contains('collapsed') ? '[+]' : '[-]';
        }}
        /* More JavaScript... */
    </script>
</head>
<body>
    <div class="header">
        <div>{model_info}</div>
        <div>{timestamp} - {message_count} messages</div>
    </div>
    <div id="controls">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
    </div>
    <div id="content">
"""
```

#### 3.2 Entry Generation Logic
```python
def _format_entry(self, fields: ChatDocLoggerFields) -> str:
    """Convert log fields to HTML entry"""
    entry_id = f"entry_{len(self.entries)}"
    entity_type = fields.responder.upper()
    
    # Build hierarchical structure
    html_parts = [f'<div class="entry {entity_type.lower()}" id="{entry_id}">']
    
    # Add entity header
    if fields.task_name and fields.task_name != "root":
        html_parts.append(f'<div class="entity-header">{fields.task_name} → {entity_type}</div>')
    else:
        html_parts.append(f'<div class="entity-header">{entity_type}</div>')
    
    # Add collapsible sections
    if fields.tool:
        html_parts.append(self._format_tool_section(fields))
    
    # Add main content
    if fields.content:
        html_parts.append(self._format_content_section(fields.content))
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)
```

#### 3.3 Tool Section Formatting
```python
def _format_tool_section(self, fields: ChatDocLoggerFields) -> str:
    """Format tool calls with proper nesting"""
    tool_id = f"tool_{self.tool_counter}"
    self.tool_counter += 1
    
    # Parse tool information
    tool_name = fields.tool
    tool_type = fields.tool_type
    
    # Build tool section HTML
    return f"""
    <div class="tool-section">
        <div class="toggle" onclick="toggle('{tool_id}')">[+]</div>
        <span class="tool-name">{tool_name}({self._format_tool_params(fields)})</span>
        <div id="{tool_id}" class="tool-content collapsed">
            <!-- Tool result and raw call details -->
        </div>
    </div>
    """
```

### 4. Integration Points

#### 4.1 Task Configuration
Add to `TaskConfig`:
```python
class TaskConfig(BaseModel):
    # ... existing fields ...
    enable_html_logging: bool = True
    html_log_dir: str = "logs"
```

#### 4.2 Logger Initialization
Modify `init_loggers` in `task.py`:
```python
def init_loggers(self, tsv_formatter: logging.Formatter | None = None) -> None:
    # ... existing logger setup ...
    
    if self.config.enable_html_logging:
        from langroid.agent.logging.html_logger import HTMLLogger
        self.html_logger = HTMLLogger(
            filename=self.name or "root",
            log_dir=self.config.html_log_dir
        )
```

#### 4.3 Message Logging
Update `log_message` method:
```python
def log_message(self, resp: ChatDocument) -> None:
    # ... existing logging ...
    
    if hasattr(self, 'html_logger') and self.html_logger:
        fields = ChatDocLoggerFields.create(resp, self.id, self.name)
        self.html_logger.log(fields)
```

### 5. Testing Strategy

#### 5.1 Unit Tests
1. Test HTML generation for various message types
2. Test proper escaping of special characters
3. Test file creation and writing
4. Test JavaScript functionality (via parsing)

#### 5.2 Integration Tests
1. Test with simple single-agent tasks
2. Test with multi-agent tasks and sub-tasks
3. Test with various tool types
4. Test with long-running conversations

#### 5.3 Manual Testing
1. Verify visual appearance matches specification
2. Test collapsible functionality in browsers
3. Test performance with large logs
4. Verify accessibility features

### 6. Implementation Timeline

1. **Phase 1**: Core HTML logger class and basic integration (2-3 hours)
2. **Phase 2**: HTML generation with proper styling (2-3 hours)
3. **Phase 3**: JavaScript functionality and interactivity (1-2 hours)
4. **Phase 4**: Testing and refinement (1-2 hours)

### 7. Key Considerations

#### 7.1 Performance
- Stream writes to avoid memory buildup
- Efficient string concatenation
- Minimal JavaScript for responsiveness

#### 7.2 Security
- Proper HTML escaping to prevent XSS
- No external dependencies (self-contained)
- Safe file path handling

#### 7.3 Compatibility
- Test across major browsers
- Ensure proper UTF-8 encoding
- Handle special characters in content

#### 7.4 Edge Cases
- Empty messages
- Very long content
- Special characters in tool names
- Malformed tool responses
- System messages without content

### 8. File Structure

```
langroid/
├── agent/
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── html_logger.py  # New file
│   │   └── ...
│   └── task.py  # Modified
└── tests/
    └── main/
        └── test_html_logger.py  # New test file
```

### 9. Future Extensions

While out of scope for initial implementation, consider:
- Configuration for color themes
- Export functionality
- Search within logs
- Performance optimizations for very large logs
- Real-time streaming updates

### 10. Success Criteria

The implementation will be considered successful when:
1. HTML logs are generated alongside existing logs
2. All log information is preserved and accessible
3. Collapsible sections work smoothly
4. Visual design matches specification
5. No performance impact on task execution
6. Tests pass and edge cases are handled