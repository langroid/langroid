"""Client Language Model that delegates to MCP client's sampling handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from fastmcp.client.sampling import SamplingMessage
from mcp.types import TextContent

from langroid.language_models.base import (
    LanguageModel,
    LLMConfig,
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    OpenAIJsonSchemaSpec,
    OpenAIToolSpec,
    Role,
    ToolChoiceTypes,
)

if TYPE_CHECKING:
    from fastmcp import Context


class ClientLMConfig(LLMConfig):
    """Configuration for MCP client-based LLM."""

    type: str = "client"
    chat_context_length: int = 1_000_000_000  # effectively infinite
    max_output_tokens: int = 1000  # reasonable default for most models


class ClientLM(LanguageModel):
    """LLM that uses MCP client's sampling handler."""

    def __init__(self, config: ClientLMConfig):
        super().__init__(config)
        self.config: ClientLMConfig = config
        self._context: Optional["Context"] = None  # Store context as instance attribute

    def set_context(self, context: "Context") -> None:
        """Set the MCP context for sampling."""
        self._context = context

    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 1000,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        """Convert Langroid messages to MCP format and use ctx.sample()."""
        if not self._context:
            raise RuntimeError("No MCP context available for sampling")

        # Convert messages - handle system, user, assistant roles
        # MCP accepts str or SamplingMessage objects
        mcp_messages: List[Union[str, SamplingMessage]] = []
        system_prompt: Optional[str] = None

        # Ensure messages is a list
        if isinstance(messages, str):
            messages = [LLMMessage(role=Role.USER, content=messages)]

        for msg in messages:
            # Use SamplingMessage with TextContent for better type safety
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            else:
                mcp_messages.append(
                    SamplingMessage(
                        role=msg.role,
                        content=TextContent(type="text", text=msg.content),
                    )
                )

        # Call MCP sampling
        temperature = self.config.temperature

        # Use FastMCP's sample method - it correctly uses snake_case parameters
        result = await self._context.sample(
            messages=mcp_messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Convert response back to Langroid format
        # MCP sample returns TextContent or ImageContent
        if isinstance(result, TextContent):
            message_text = result.text
        else:
            # ImageContent case - for now, we don't support image responses
            raise NotImplementedError(
                "ClientLM does not currently support ImageContent responses "
                "from MCP sampling"
            )

        return LLMResponse(
            message=message_text,
            cached=False,
        )

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 1000,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        """Synchronous chat is not implemented for ClientLM."""
        raise NotImplementedError(
            "ClientLM only supports async operations. Use achat() instead."
        )

    async def agenerate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """Generate completion from a prompt using MCP sampling."""
        # Convert prompt to messages format for consistency
        return await self.achat(prompt, max_tokens=max_tokens)

    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """Synchronous generate is not implemented for ClientLM."""
        raise NotImplementedError(
            "ClientLM only supports async operations. Use agenerate() instead."
        )

    def get_stream(self) -> bool:
        """ClientLM does not support streaming."""
        return False

    def set_stream(self, stream: bool) -> bool:
        """ClientLM does not support streaming."""
        return False
