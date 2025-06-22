"""Example client that connects to Langroid MCP server with sampling handler.

This example demonstrates how to use Langroid agents via MCP (Model Context Protocol).
The Langroid MCP server exposes Langroid agents as tools that can be used by any MCP client.

Prerequisites:
1. Set your LLM API key (either as environment variable or in .env file):
   - For OpenAI: OPENAI_API_KEY="your-api-key"
   - For Anthropic: ANTHROPIC_API_KEY="your-api-key"
   
   You can either export these as environment variables or create a .env file
   in your project root with these values.

2. Install required dependencies:
   pip install openai anthropic fastmcp python-dotenv

How to run:
1. This example automatically starts the Langroid MCP server via UVX
2. Run the example:
   python examples/mcp/langroid_agent_client.py

Optional: Set LLM provider (defaults to OpenAI):
   export LLM_PROVIDER=anthropic  # to use Claude instead of GPT-4

The example demonstrates:
- Basic chat interactions with Langroid agents
- Using custom agent names
- Enabling tools like web search
- Multi-turn task-based conversations
"""

import asyncio
import os
from typing import List
from dotenv import load_dotenv
from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient
from fastmcp.client.transports import StdioTransport
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext
from mcp.types import TextContent
import openai

# Load environment variables from .env file
load_dotenv()


# Initialize your LLM client
openai_client = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
)


async def openai_sampling_handler(
    messages: List[SamplingMessage],
    params: SamplingParams,
    context: RequestContext,
) -> str:
    """Handle sampling requests by calling OpenAI."""
    # Convert MCP messages to OpenAI format
    openai_messages = []

    # Handle system prompt if provided
    if params and params.systemPrompt:
        openai_messages.append({"role": "system", "content": params.systemPrompt})

    for msg in messages:
        if isinstance(msg, str):
            openai_messages.append({"role": "user", "content": msg})
        elif isinstance(msg, dict):
            openai_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        else:
            # Handle SamplingMessage objects
            content = msg.content
            if isinstance(content, TextContent):
                content = content.text
            openai_messages.append({"role": msg.role, "content": content})

    # Call OpenAI
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=openai_messages,
            temperature=params.temperature if params else 0.7,
            max_tokens=params.maxTokens if params else 1000,
        )

        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"


async def anthropic_sampling_handler(
    messages: List[SamplingMessage],
    params: SamplingParams,
    context: RequestContext,
) -> str:
    """Alternative: Handle sampling requests using Anthropic Claude."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
    )

    # Convert to Anthropic format
    anthropic_messages = []
    system_prompt = ""

    if params and params.systemPrompt:
        system_prompt = params.systemPrompt

    for msg in messages:
        if isinstance(msg, str):
            anthropic_messages.append({"role": "user", "content": msg})
        elif isinstance(msg, dict):
            anthropic_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        else:
            content = msg.content
            if isinstance(content, TextContent):
                content = content.text
            anthropic_messages.append({"role": msg.role, "content": content})

    try:
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=anthropic_messages,
            system=system_prompt,
            temperature=params.temperature if params else 0.7,
            max_tokens=params.maxTokens if params else 1000,
        )

        return response.content[0].text
    except Exception as e:
        return f"Error calling Anthropic: {str(e)}"


async def main():
    """Main function demonstrating Langroid MCP server usage."""

    print("Starting Langroid MCP Client Example...")
    print("Make sure the Langroid MCP server is running!")
    print("You can start it with: uvx --from langroid langroid-mcp-server")
    print()

    # Connect to Langroid MCP server
    transport = StdioTransport(
        command="uvx", args=["--from", "langroid", "langroid-mcp-server"]
    )

    # Choose your LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if llm_provider == "anthropic":
        sampling_handler = anthropic_sampling_handler
        print("Using Anthropic Claude as LLM provider")
    else:
        sampling_handler = openai_sampling_handler
        print("Using OpenAI as LLM provider")

    try:
        async with FastMCPClient(
            transport, sampling_handler=sampling_handler
        ) as client:
            print("\nConnected to Langroid MCP server!")

            # Get available tools
            tool_classes = await client.get_tools_async()
            tool_names = [cls.__name__ for cls in tool_classes]
            print(f"\nAvailable tools: {tool_names}")

            # Get the langroid_chat tool
            LangroidChat = await client.get_tool_async("langroid_chat")

            # Example 1: Basic chat
            print("\n--- Example 1: Basic Chat ---")
            chat_tool = LangroidChat(message="What is the capital of France?")
            response = await chat_tool.call_tool_async()
            print(f"Response: {response}")

            # Example 2: Chat with custom agent name
            print("\n--- Example 2: Custom Agent Name ---")
            chat_tool = LangroidChat(
                message="Tell me a fun fact about AI", agent_name="AIExpert"
            )
            response = await chat_tool.call_tool_async()
            print(f"Response: {response}")

            # Example 3: Chat with web search tool
            print("\n--- Example 3: With Web Search ---")
            chat_tool = LangroidChat(
                message="Search for the latest news about large language models",
                enable_tools=["web_search"],
            )
            response = await chat_tool.call_tool_async()
            print(f"Response: {response}")

            # Get the langroid_task tool for multi-turn conversations
            LangroidTask = await client.get_tool_async("langroid_task")

            # Example 4: Task-based interaction
            print("\n--- Example 4: Task-based Interaction ---")
            task_tool = LangroidTask(
                message="Help me write a Python function to calculate fibonacci numbers",
                max_turns=3,
            )
            response = await task_tool.call_tool_async()
            print(f"Response: {response}")

            # Example 5: Task with tools
            print("\n--- Example 5: Task with Tools ---")
            task_tool = LangroidTask(
                message="Research and summarize information about quantum computing",
                enable_tools=["web_search"],
                agent_name="ResearchAgent",
                max_turns=5,
            )
            response = await task_tool.call_tool_async()
            print(f"Response: {response}")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. The Langroid MCP server is installed and running")
        print("2. Your LLM API keys are set in environment variables")
        print(
            "3. You have installed required dependencies (openai, anthropic, fastmcp)"
        )


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
