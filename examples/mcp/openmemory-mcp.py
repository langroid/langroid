"""
Example script for connecting to OpenMemory MCP server.

OpenMemory MCP provides a memory-aware MCP server with endpoints for storing and retrieving memories.
This example shows how to connect to an OpenMemory MCP server using the DualEndpointTransport.

Run like this:

    uv run examples/mcp/openmemory-mcp.py --model gpt-4.1-mini

For setup instructions for OpenMemory, see:
https://mem0.ai/blog/how-to-make-your-clients-more-context-aware-with-openmemory-mcp/
"""

import os
import datetime
import logging
from typing import Optional

import langroid as lr
import langroid.language_models as lm
from langroid.mytypes import NonToolAction
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.agent.tools.mcp.dual_endpoint import DualEndpointTransport
from fire import Fire

# Set up logging
logging.basicConfig(level=logging.INFO)


# Removed the create_openmemory_transport function since we're now handling URL patterns directly in main()


# We're now handling the tool connection directly in main() for better error handling


async def main(
    model: str = "",
    base_url: str = "http://localhost:8765",
    client_name: str = "langroid",
    message_path: Optional[str] = None,
    retry_urls: bool = True,
):
    """
    Connect to OpenMemory MCP and run a conversation with available tools.

    Args:
        model: LLM model to use (defaults to gpt-4.1-mini if empty)
        base_url: URL of the OpenMemory server
        client_name: Client name identifier for OpenMemory
        message_path: Optional custom path for message posting endpoint
    """
    # Get the current user ID from environment
    user_id = os.getenv("USER", "default_user")

    print(f"Connecting to OpenMemory at {base_url} as user {user_id}...")

    print("Setting up OpenMemory transport...")

    # If retry_urls is True, we'll try multiple different URL patterns
    url_patterns = []
    if retry_urls:
        # Try multiple URL patterns for the SSE endpoint
        url_patterns = [
            # Pattern 1: Standard
            (
                f"{base_url}/mcp/{client_name}/sse/{user_id}",
                f"{base_url}/mcp/{client_name}/sse/{user_id}/messages/",
            ),
            # Pattern 2: With API prefix
            (
                f"{base_url}/api/mcp/{client_name}/sse/{user_id}",
                f"{base_url}/api/mcp/{client_name}/sse/{user_id}/messages/",
            ),
            # Pattern 3: Different structure
            (f"{base_url}/sse/{user_id}", f"{base_url}/messages/"),
            # Pattern 4: Directly at base URL with client in headers
            (f"{base_url}/sse", f"{base_url}/messages"),
            # Pattern 5: Try port 3000 instead
            (
                f"http://localhost:3000/mcp/{client_name}/sse/{user_id}",
                f"http://localhost:3000/mcp/{client_name}/sse/{user_id}/messages/",
            ),
            # Pattern 6: Try port 3000 with API prefix
            (
                f"http://localhost:3000/api/mcp/{client_name}/sse/{user_id}",
                f"http://localhost:3000/api/mcp/{client_name}/sse/{user_id}/messages/",
            ),
            # Pattern 7: Try port 8080
            (
                f"http://localhost:8080/mcp/{client_name}/sse/{user_id}",
                f"http://localhost:8080/mcp/{client_name}/sse/{user_id}/messages/",
            ),
            # Pattern 8: Try port 8080 with API prefix
            (
                f"http://localhost:8080/api/mcp/{client_name}/sse/{user_id}",
                f"http://localhost:8080/api/mcp/{client_name}/sse/{user_id}/messages/",
            ),
            # Pattern 9: Try port 3001
            (
                f"http://localhost:3001/mcp/{client_name}/sse/{user_id}",
                f"http://localhost:3001/mcp/{client_name}/sse/{user_id}/messages/",
            ),
        ]
    else:
        # Just use the standard pattern
        url_patterns = [
            (
                f"{base_url}/mcp/{client_name}/sse/{user_id}",
                f"{base_url}/mcp/{client_name}/sse/{user_id}/messages/",
            )
        ]

    all_tools = None
    last_error = None

    # Try each URL pattern
    for sse_url, msg_url in url_patterns:
        print(f"\nTrying SSE URL: {sse_url}")
        print(f"Trying message URL: {msg_url}")

        # Create custom headers with client name
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Client-Name": client_name,
        }

        # Create transport with this URL pattern
        transport = DualEndpointTransport(
            sse_url=sse_url,
            message_url=msg_url,
            headers=headers,
            sse_read_timeout=10,  # 10 second timeout
        )

        print("Attempting to retrieve tools from OpenMemory server...")

        try:
            # Set a short timeout for testing
            all_tools = await get_tools_async(
                transport, read_timeout_seconds=datetime.timedelta(seconds=10)
            )
            print("Successfully retrieved tools!")
            # If we get here, we found a working URL pattern
            break
        except Exception as e:
            last_error = str(e)
            print(f"Error retrieving tools: {e}")
            print("Trying next URL pattern...")

    if all_tools is None:
        print(
            "Failed to connect to OpenMemory MCP server after trying all URL patterns."
        )
        print(f"Last error: {last_error}")
        print("\nTroubleshooting suggestions:")
        print("1. Make sure the OpenMemory server is running (docker-compose up)")
        print("2. Check for processes using required ports (3000, 8765, 8080, etc)")
        print("3. Verify your OpenMemory installation and configuration")
        print("4. Check OpenMemory server logs for any errors")
        return

    print(f"Found {len(all_tools)} tools from OpenMemory MCP")

    # Create a Langroid agent with the tools
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )

    # Enable all available tools
    agent.enable_message(all_tools)

    # Create and run the task
    task = lr.Task(agent, interactive=False)
    await task.run_async(
        "Based on the TOOLs available to you, greet the user and "
        "tell them what kinds of help you can provide with memory management."
        "\n\nThen, wait for user input to interact with the OpenMemory system."
    )


if __name__ == "__main__":
    Fire(main)
