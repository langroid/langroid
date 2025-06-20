"""Client Language Model that delegates to MCP client's sampling handler."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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
    context: Optional[Any] = None  # Set by MCP server at runtime
    chat_context_length: int = 1_000_000_000  # effectively infinite


class ClientLM(LanguageModel):
    """LLM that uses MCP client's sampling handler."""

    def __init__(self, config: ClientLMConfig):
        super().__init__(config)
        self.config: ClientLMConfig = config

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
        if not self.config.context:
            raise RuntimeError("No MCP context available for sampling")

        # Convert messages - handle system, user, assistant roles
        mcp_messages = []
        system_prompt = None

        # Ensure messages is a list
        if isinstance(messages, str):
            messages = [LLMMessage(role=Role.USER, content=messages)]

        for msg in messages:
            if msg.role == "system":
                # MCP expects system prompt separately
                system_prompt = msg.content
            else:
                mcp_messages.append({"role": msg.role, "content": msg.content})

        # Call MCP sampling
        temperature = self.config.temperature
        result = await self.config.context.sample(
            messages=mcp_messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Convert response back to Langroid format
        # MCP sample returns TextContent or ImageContent
        message_text = ""
        if hasattr(result, "text"):
            message_text = result.text
        elif hasattr(result, "content"):
            message_text = result.content
        else:
            message_text = str(result)

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
