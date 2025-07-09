"""HTML Logger for Langroid Task System.

This module provides an HTML logger that creates self-contained HTML files
with collapsible log entries for better visualization of agent interactions.
"""

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langroid.pydantic_v1 import BaseModel
from langroid.utils.logging import setup_logger


class HTMLLogger:
    """Logger that outputs task logs as interactive HTML files."""

    def __init__(
        self,
        filename: str,
        log_dir: str = "logs",
        model_info: str = "",
        append: bool = False,
    ):
        """Initialize the HTML logger.

        Args:
            filename: Base name for the log file (without extension)
            log_dir: Directory to store log files
            model_info: Information about the model being used
            append: Whether to append to existing file
        """
        self.filename = filename
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.log_dir / f"{filename}.html"
        self.model_info = model_info
        self.entries: List[Dict[str, Any]] = []
        self.entry_counter = 0
        self.tool_counter = 0

        # Logger for errors
        self.logger = setup_logger(__name__)

        if not append or not self.file_path.exists():
            self._write_header()

    def _write_header(self) -> None:
        """Write the HTML header with CSS and JavaScript."""
        timestamp = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="2">
    <title>{self.filename} - Langroid Task Log</title>
    <style>
        body {{
            background-color: #1e1e1e;
            color: #f0f0f0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .header {{
            border: 2px solid #d4a017;
            padding: 15px;
            margin-bottom: 20px;
            color: #d4a017;
            background-color: #2b2b2b;
            border-radius: 5px;
        }}
        
        .header-line {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .separator {{
            border-bottom: 2px solid #d4a017;
            margin: 20px 0;
        }}
        
        .controls {{
            margin-bottom: 20px;
        }}
        
        .controls {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .controls button {{
            background-color: #333;
            color: #f0f0f0;
            border: 1px solid #555;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
        }}
        
        .controls button:hover {{
            background-color: #444;
            border-color: #d4a017;
        }}
        
        .controls label {{
            color: #f0f0f0;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }}
        
        .controls input[type="checkbox"] {{
            cursor: pointer;
        }}
        
        .hidden {{
            display: none !important;
        }}
        
        .entry {{
            margin-bottom: 15px;
            padding-left: 10px;
        }}
        
        .entry.faded {{
            opacity: 0.4;
        }}
        
        .entry.important {{
            opacity: 1.0;
        }}
        
        .entry.user .entity-header {{
            color: #00bfff;
        }}
        
        .entry.assistant .entity-header {{
            color: #ff6b6b;
        }}
        
        .entry.llm .entity-header {{
            color: #00ff00;
        }}
        
        .entry.agent .entity-header {{
            color: #ff9500;
        }}
        
        .entry.system .entity-header {{
            color: #888;
        }}
        
        .entry.other .entity-header {{
            color: #999;
        }}
        
        .entity-header {{
            font-weight: bold;
            margin-bottom: 5px;
            cursor: pointer;
        }}
        
        .entity-header:hover {{
            opacity: 0.8;
        }}
        
        .header-main {{
            /* Removed text-transform to preserve tool name casing */
            display: inline;
        }}
        
        .header-content {{
            margin-left: 30px;
            opacity: 0.7;
            font-weight: normal;
            font-style: italic;
            display: block;
        }}
        
        .entry-content {{
            margin-left: 20px;
            margin-top: 5px;
        }}
        
        .entry-content.collapsed {{
            display: none;
        }}
        
        .collapsible {{
            margin: 5px 0;
            margin-left: 20px;
        }}
        
        .toggle {{
            cursor: pointer;
            user-select: none;
            color: #00ff00;
            display: inline-block;
            width: 25px;
            font-family: monospace;
            margin-right: 5px;
        }}
        
        .toggle:hover {{
            color: #00ff00;
            text-shadow: 0 0 5px #00ff00;
        }}
        
        .content {{
            margin-left: 25px;
            margin-top: 5px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .main-content {{
            margin-top: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .collapsed .content {{
            display: none;
        }}
        
        .tool-section {{
            margin: 10px 0;
            margin-left: 20px;
        }}
        
        .tool-name {{
            color: #d4a017;
            font-weight: bold;
        }}
        
        .tool-result {{
            margin-left: 25px;
        }}
        
        .tool-result.success {{
            color: #00ff00;
        }}
        
        .tool-result.error {{
            color: #ff0000;
        }}
        
        .code-block {{
            background-color: #2b2b2b;
            border: 1px solid #444;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            overflow-x: auto;
        }}
        
        .metadata {{
            color: #888;
            font-size: 0.9em;
            margin-left: 25px;
        }}
        
        
        pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
    <script>
        function toggleEntry(entryId) {{
            const contentElement = document.getElementById(entryId + '_content');
            const toggleElement = document.querySelector(
                '#' + entryId + ' .entity-header .toggle'
            );
            
            if (!contentElement || !toggleElement) return;
            
            if (contentElement.classList.contains('collapsed')) {{
                contentElement.classList.remove('collapsed');
                toggleElement.textContent = '[-]';
                // Save expanded state
                localStorage.setItem('expanded_' + entryId, 'true');
            }} else {{
                contentElement.classList.add('collapsed');
                toggleElement.textContent = '[+]';
                // Save collapsed state
                localStorage.setItem('expanded_' + entryId, 'false');
            }}
        }}
        
        function toggle(id) {{
            const element = document.getElementById(id);
            if (!element) return;
            
            element.classList.toggle('collapsed');
            const toggle = element.querySelector('.toggle');
            if (toggle) {{
                toggle.textContent = element.classList.contains('collapsed') 
                    ? '[+]' : '[-]';
            }}
            
            // Save collapsed state for collapsible sections
            localStorage.setItem(
                'collapsed_' + id, element.classList.contains('collapsed')
            );
        }}
        
        let allExpanded = false;
        
        function toggleAll() {{
            const btn = document.getElementById('toggleAllBtn');
            if (allExpanded) {{
                collapseAll();
                btn.textContent = 'Expand All';
                allExpanded = false;
            }} else {{
                expandAll();
                btn.textContent = 'Collapse All';
                allExpanded = true;
            }}
        }}
        
        function expandAll() {{
            // Expand all visible main entries
            const entries = document.querySelectorAll(
                '.entry:not(.hidden) .entry-content'
            );
            entries.forEach(element => {{
                element.classList.remove('collapsed');
            }});
            
            // Update all visible main entry toggles
            const entryToggles = document.querySelectorAll(
                '.entry:not(.hidden) .entity-header .toggle'
            );
            entryToggles.forEach(toggle => {{
                toggle.textContent = '[-]';
            }});
            
            // Expand all visible sub-sections
            const collapsibles = document.querySelectorAll(
                '.entry:not(.hidden) .collapsible'
            );
            collapsibles.forEach(element => {{
                element.classList.remove('collapsed');
                const toggle = element.querySelector('.toggle');
                if (toggle) {{
                    toggle.textContent = '[-]';
                }}
            }});
        }}
        
        function collapseAll() {{
            // Collapse all visible entries
            const entries = document.querySelectorAll(
                '.entry:not(.hidden) .entry-content'
            );
            entries.forEach(element => {{
                element.classList.add('collapsed');
            }});
            
            // Update all visible entry toggles
            const entryToggles = document.querySelectorAll(
                '.entry:not(.hidden) .entity-header .toggle'
            );
            entryToggles.forEach(toggle => {{
                toggle.textContent = '[+]';
            }});
            
            // Collapse all visible sub-sections
            const collapsibles = document.querySelectorAll(
                '.entry:not(.hidden) .collapsible'
            );
            collapsibles.forEach(element => {{
                element.classList.add('collapsed');
                const toggle = element.querySelector('.toggle');
                if (toggle) {{
                    toggle.textContent = '[+]';
                }}
            }});
        }}
        
        function filterEntries() {{
            const checkbox = document.getElementById('filterCheckbox');
            const entries = document.querySelectorAll('.entry');
            
            // Save checkbox state to localStorage
            localStorage.setItem('filterImportant', checkbox.checked);
            
            if (checkbox.checked) {{
                // Show only important entries
                entries.forEach(entry => {{
                    const isImportant = entry.classList.contains('important');
                    if (isImportant) {{
                        entry.classList.remove('hidden');
                    }} else {{
                        entry.classList.add('hidden');
                    }}
                }});
            }} else {{
                // Show all entries
                entries.forEach(entry => {{
                    entry.classList.remove('hidden');
                }});
            }}
            
            // Reset toggle button state
            allExpanded = false;
            document.getElementById('toggleAllBtn').textContent = 'Expand All';
        }}
        
        // Initialize all as collapsed on load
        document.addEventListener('DOMContentLoaded', function() {{
            collapseAll();
            
            // Restore checkbox state from localStorage
            const checkbox = document.getElementById('filterCheckbox');
            const savedState = localStorage.getItem('filterImportant');
            if (savedState !== null) {{
                // Use saved state if it exists
                checkbox.checked = savedState === 'true';
            }}
            // Apply filter based on checkbox state (default is checked)
            if (checkbox.checked) {{
                filterEntries();
            }}
            
            // Restore expanded states from localStorage
            const entries = document.querySelectorAll('.entry');
            entries.forEach(entry => {{
                const entryId = entry.id;
                const expandedState = localStorage.getItem('expanded_' + entryId);
                if (expandedState === 'true') {{
                    const contentElement = document.getElementById(
                        entryId + '_content'
                    );
                    const toggleElement = entry.querySelector('.entity-header .toggle');
                    if (contentElement && toggleElement) {{
                        contentElement.classList.remove('collapsed');
                        toggleElement.textContent = '[-]';
                    }}
                }}
            }});
            
            // Restore collapsible section states
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(collapsible => {{
                const id = collapsible.id;
                const collapsedState = localStorage.getItem('collapsed_' + id);
                if (collapsedState === 'false') {{
                    collapsible.classList.remove('collapsed');
                    const toggle = collapsible.querySelector('.toggle');
                    if (toggle) {{
                        toggle.textContent = '[-]';
                    }}
                }}
            }});
        }});
    </script>
</head>
<body>
    <div class="header">
        <div class="header-line">
            <div>{self.filename}</div>
            <div id="timestamp">{timestamp}</div>
        </div>
    </div>
    
    <div class="separator"></div>
    
    <div class="controls">
        <button id="toggleAllBtn" onclick="toggleAll()">Expand All</button>
        <label style="margin-left: 20px;">
            <input type="checkbox" id="filterCheckbox" 
                   onchange="filterEntries()" checked>
            Show only important responses
        </label>
    </div>
    
    <div id="content">
"""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            self.logger.error(f"Failed to write HTML header: {e}")

    def log(self, fields: BaseModel) -> None:
        """Log a message entry.

        Args:
            fields: ChatDocLoggerFields containing all log information
        """
        try:
            entry_html = self._format_entry(fields)
            self._append_to_file(entry_html)
            self.entry_counter += 1
        except Exception as e:
            self.logger.error(f"Failed to log entry: {e}")

    def _format_entry(self, fields: BaseModel) -> str:
        """Format a log entry as HTML.

        Args:
            fields: ChatDocLoggerFields containing all log information

        Returns:
            HTML string for the entry
        """
        entry_id = f"entry_{self.entry_counter}"

        # Get all relevant fields
        responder = str(getattr(fields, "responder", "UNKNOWN"))
        task_name = getattr(fields, "task_name", "root")
        # TODO (CLAUDE) display sender_entity in parens right after responder,
        # other than LLM, e.g. AGENT (USER)
        sender_entity = str(getattr(fields, "sender_entity", ""))
        tool = getattr(fields, "tool", "")
        tool_type = getattr(fields, "tool_type", "")
        content = getattr(fields, "content", "")
        recipient = getattr(fields, "recipient", "")

        # Determine CSS class based on responder
        responder_upper = responder.upper()
        if "USER" in responder_upper:
            css_class = "user"
        elif "LLM" in responder_upper:
            css_class = "llm"
        elif "AGENT" in responder_upper:
            css_class = "agent"
        elif "SYSTEM" in responder_upper:
            css_class = "system"
        else:
            css_class = "other"

        # Determine opacity class based on mark
        mark = getattr(fields, "mark", "")
        opacity_class = "important" if mark == "*" else "faded"

        # Start building the entry
        html_parts = [
            f'<div class="entry {css_class} {opacity_class}" id="{entry_id}">'
        ]

        # Build smart header
        entity_parts = []  # Main header line with entity info
        content_preview = ""  # Second line with content preview

        # Add task name if not root
        if task_name and task_name != "root":
            entity_parts.append(task_name)

        # Handle different responder types
        if "USER" in responder_upper:
            # Add responder with sender_entity in parens if different
            if sender_entity and sender_entity != responder:
                entity_parts.append(f"USER ({sender_entity})")
            else:
                entity_parts.append("USER")
            # Show user input preview on second line
            if content:
                preview = content.replace("\n", " ")[:60]
                if len(content) > 60:
                    preview += "..."
                content_preview = f'"{preview}"'

        elif "LLM" in responder_upper:
            # Get model info from instance - don't uppercase it
            model_label = "LLM"
            if self.model_info:
                model_label = f"LLM ({self.model_info})"

            if tool and tool_type:
                # LLM making a tool call - don't uppercase tool names
                entity_parts.append(f"{model_label} → {tool_type}[{tool}]")
            else:
                # LLM generating plain text response
                entity_parts.append(model_label)
                if content:
                    # Show first line or first 60 chars on second line
                    first_line = content.split("\n")[0].strip()
                    if first_line:
                        preview = first_line[:60]
                        if len(first_line) > 60:
                            preview += "..."
                        content_preview = f'"{preview}"'

        elif "AGENT" in responder_upper:
            # Add responder with sender_entity in parens if different
            agent_label = "AGENT"
            if sender_entity and sender_entity != responder:
                agent_label = f"AGENT ({sender_entity})"

            # Agent responding (usually tool handling)
            if tool:
                entity_parts.append(f"{agent_label}[{tool}]")
                # Show tool result preview on second line if available
                if content:
                    preview = content.replace("\n", " ")[:40]
                    if len(content) > 40:
                        preview += "..."
                    content_preview = f"→ {preview}"
            else:
                entity_parts.append(agent_label)
                if content:
                    preview = content[:50]
                    if len(content) > 50:
                        preview += "..."
                    content_preview = f'"{preview}"'

        elif "SYSTEM" in responder_upper:
            entity_parts.append("SYSTEM")
            if content:
                preview = content[:50]
                if len(content) > 50:
                    preview += "..."
                content_preview = f'"{preview}"'
        else:
            # Other responder types (like Task)
            entity_parts.append(responder)

        # Add recipient info if present
        if recipient:
            entity_parts.append(f"→ {recipient}")

        # Construct the two-line header
        header_main = " ".join(entity_parts)

        # Build the header HTML with toggle, mark, and main content on same line
        header_html = '<span class="toggle">[+]</span> '

        # Note: opacity_class already determined above

        # Add the main header content
        header_html += f'<span class="header-main">{html.escape(header_main)}</span>'

        # Add preview on second line if present
        if content_preview:
            header_html += (
                f'\n    <div class="header-content">'
                f"{html.escape(content_preview)}</div>"
            )

        # Add expandable header
        html_parts.append(
            f"""
<div class="entity-header" onclick="toggleEntry('{entry_id}')">
    {header_html}
</div>
<div id="{entry_id}_content" class="entry-content collapsed">"""
        )

        # Add collapsible sections

        # System messages (if any)
        system_content = self._extract_system_content(fields)
        if system_content:
            for idx, (label, content) in enumerate(system_content):
                section_id = f"{entry_id}_system_{idx}"
                html_parts.append(
                    self._create_collapsible_section(section_id, label, content)
                )

        # Tool information
        tool = getattr(fields, "tool", None)
        # Only add tool section if tool exists and is not empty
        if tool and tool.strip():
            tool_html = self._format_tool_section(fields, entry_id)
            html_parts.append(tool_html)

        # Main content
        content = getattr(fields, "content", "")
        if content and not (
            tool and tool.strip()
        ):  # Don't duplicate content if it's a tool
            html_parts.append(f'<div class="main-content">{html.escape(content)}</div>')

        # Metadata (recipient, blocked)
        metadata_parts = []
        recipient = getattr(fields, "recipient", None)
        if recipient:
            metadata_parts.append(f"Recipient: {recipient}")

        block = getattr(fields, "block", None)
        if block:
            metadata_parts.append(f"Blocked: {block}")

        if metadata_parts:
            html_parts.append(
                f'<div class="metadata">{" | ".join(metadata_parts)}</div>'
            )

        # Close entry content div
        html_parts.append("</div>")  # Close entry-content
        html_parts.append("</div>")  # Close entry
        return "\n".join(html_parts)

    def _extract_system_content(self, fields: BaseModel) -> List[tuple[str, str]]:
        """Extract system-related content from fields.

        Returns:
            List of (label, content) tuples
        """
        system_content = []

        # Check for common system message patterns in content
        content = getattr(fields, "content", "")
        if content:
            # Look for patterns like "[System Prompt]" or "System Reminder:"
            if "[System Prompt]" in content or "System Prompt" in content:
                system_content.append(("System Prompt", content))
            elif "[System Reminder]" in content or "System Reminder" in content:
                system_content.append(("System Reminder", content))

        return system_content

    def _create_collapsible_section(
        self, section_id: str, label: str, content: str
    ) -> str:
        """Create a collapsible section.

        Args:
            section_id: Unique ID for the section
            label: Label to display
            content: Content to show when expanded

        Returns:
            HTML string for the collapsible section
        """
        return f"""
<div class="collapsible collapsed" id="{section_id}">
    <span class="toggle" onclick="toggle('{section_id}')">[+]</span> {label}
    <div class="content">{html.escape(content)}</div>
</div>"""

    def _format_tool_section(self, fields: BaseModel, entry_id: str) -> str:
        """Format tool-related information.

        Args:
            fields: ChatDocLoggerFields containing tool information
            entry_id: Parent entry ID

        Returns:
            HTML string for the tool section
        """
        tool = getattr(fields, "tool", "")
        tool_type = getattr(fields, "tool_type", "")
        content = getattr(fields, "content", "")

        tool_id = f"{entry_id}_tool_{self.tool_counter}"
        self.tool_counter += 1

        # Try to parse content as JSON for better formatting
        try:
            if content.strip().startswith("{"):
                content_dict = json.loads(content)
                formatted_content = json.dumps(content_dict, indent=2)
                content_html = (
                    f'<pre class="code-block">{html.escape(formatted_content)}</pre>'
                )
            else:
                content_html = html.escape(content)
        except Exception:
            content_html = html.escape(content)

        # Build tool section
        tool_name = f"{tool_type}({tool})" if tool_type else tool

        return f"""
<div class="tool-section">
    <div class="collapsible collapsed" id="{tool_id}">
        <span class="toggle" onclick="toggle('{tool_id}')">[+]</span>
        <span class="tool-name">{html.escape(tool_name)}</span>
        <div class="content">{content_html}</div>
    </div>
</div>"""

    def _append_to_file(self, content: str) -> None:
        """Append content to the HTML file.

        Args:
            content: HTML content to append
        """
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(content + "\n")
                f.flush()
        except Exception as e:
            self.logger.error(f"Failed to append to file: {e}")

    def close(self) -> None:
        """Close the HTML file with footer."""
        footer = """
    </div>
    <script>
        // Update message count
        const header = document.querySelector('.header-line div:last-child');
        if (header) {
            const messageCount = document.querySelectorAll('.entry').length;
            header.textContent = header.textContent.replace(
                /\\d+ messages/, messageCount + ' messages'
            );
        }
    </script>
</body>
</html>"""
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(footer)
        except Exception as e:
            self.logger.error(f"Failed to write HTML footer: {e}")
