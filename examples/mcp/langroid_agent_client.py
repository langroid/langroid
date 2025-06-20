"""Example client that connects to Langroid MCP server with sampling handler."""

import asyncio
import os
from typing import List, Dict, Any
from fastmcp.client import FastMCPClient
from fastmcp.client.transports import StdioTransport
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext
import openai


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
    if params and params.system_prompt:
        openai_messages.append({"role": "system", "content": params.system_prompt})

    for msg in messages:
        if isinstance(msg, str):
            openai_messages.append({"role": "user", "content": msg})
        elif isinstance(msg, dict):
            openai_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        else:
            # Handle SamplingMessage objects
            openai_messages.append({"role": msg.role, "content": msg.content})

    # Call OpenAI
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=openai_messages,
            temperature=params.temperature if params else 0.7,
            max_tokens=params.max_tokens if params else 1000,
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

    if params and params.system_prompt:
        system_prompt = params.system_prompt

    for msg in messages:
        if isinstance(msg, str):
            anthropic_messages.append({"role": "user", "content": msg})
        elif isinstance(msg, dict):
            anthropic_messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )
        else:
            anthropic_messages.append({"role": msg.role, "content": msg.content})

    try:
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=anthropic_messages,
            system=system_prompt,
            temperature=params.temperature if params else 0.7,
            max_tokens=params.max_tokens if params else 1000,
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
            tools = await client.list_tools()
            print(f"\nAvailable tools: {[tool['name'] for tool in tools]}")

            # Get the langroid_chat tool
            langroid_chat = await client.call_tool("langroid_chat")

            # Example 1: Basic chat
            print("\n--- Example 1: Basic Chat ---")
            response = await langroid_chat(
                message="What is the capital of France?",
            )
            print(f"Response: {response}")

            # Example 2: Chat with custom agent name
            print("\n--- Example 2: Custom Agent Name ---")
            response = await langroid_chat(
                message="Tell me a fun fact about AI", agent_name="AIExpert"
            )
            print(f"Response: {response}")

            # Example 3: Chat with web search tool
            print("\n--- Example 3: With Web Search ---")
            response = await langroid_chat(
                message="Search for the latest news about large language models",
                enable_tools=["web_search"],
            )
            print(f"Response: {response}")

            # Get the langroid_task tool for multi-turn conversations
            langroid_task = await client.call_tool("langroid_task")

            # Example 4: Task-based interaction
            print("\n--- Example 4: Task-based Interaction ---")
            response = await langroid_task(
                message="Help me write a Python function to calculate fibonacci numbers",
                max_turns=3,
            )
            print(f"Response: {response}")

            # Example 5: Task with tools
            print("\n--- Example 5: Task with Tools ---")
            response = await langroid_task(
                message="Research and summarize information about quantum computing",
                enable_tools=["web_search"],
                agent_name="ResearchAgent",
                max_turns=5,
            )
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
