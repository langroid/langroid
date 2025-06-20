"""
AgentTool: A tool that allows agents to spawn sub-agents with specific tools enabled.
"""

from typing import List, Optional

import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.pydantic_v1 import Field


class AgentTool(ToolMessage):
    """
    Tool that spawns a sub-agent with specified tools to handle a task.
    """

    request: str = "agent_tool"
    purpose: str = """
        <HowToUse>
        Use this tool to delegate a task to a sub-agent with specific tools enabled.
        The sub-agent will be created with the specified tools and will run the task
        non-interactively. Here is how to set the fields:
        
        - `system_message`: 
        

    """

    # Parameters for the agent tool

    system_message: Optional[str] = Field(
        ...,
        description="""
        Optional system message to configure the sub-agent's general behavior.
            A good system message will have these components:
            - Inform the sub-agent of its role, e.g. "You are a financial analyst."
            - Clear spec of the task
            - Any additional general context needed for the task, such as a
              (part of a) document, or data items, etc.
            - Specify when to use certain tools, e.g. 
                "You MUST use the 'stock_data' tool to extract stock information.
        """,
    )

    prompt: str = Field(
        ...,
        description="""
            The prompt to run the sub-agent with. This differs from the agent's
            system message: Whereas the system message configures the sub-agent's
            general role and task, the prompt is the specific input that the sub-agent
            will process. In LLM terms, the system message is sent to the LLM as
            the first message, with role = "system" or "developer", and 
            the prompt is sent as a message with role = "user".
            EXAMPLE: system_message = "You are a financial analyst, when the 
                user asks about the share-price of a company, 
                you must use your tools to do the research, and 
                return the final answer to the user."
            
            prompt = "What is the share-price of Apple Inc.?"
            """,
    )

    tools: List[str] = Field(
        ...,
        description="""
        A list of tool names to enable for the sub-agent.
        This must be a list of strings referring to the names of tools
        that are known to you. 
        If you want to enable all tools, you can set this field
         to a singleton list containing 'ALL'
        To disable all tools, set it to a singleton list containing 'NONE'
        """,
    )
    model: str = Field(
        default=None,
        description="""
            Optional name of the LLM model to use for the sub-agent, e.g. 'gpt-4.1'
            If omitted, the sub-agent will use the same model as yours.
            """,
    )
    max_iterations: Optional[int] = Field(
        default=None,
        description="Optional max iterations for the sub-agent to run the task",
    )

    def handle(self, agent: ChatAgent) -> Optional[ChatDocument]:
        """
        Handle the AgentTool by creating a sub-agent with specified tools
        and running the task non-interactively.

        Args:
            agent: The parent ChatAgent that is handling this tool
        """
        # Create chat agent config with system message if provided
        # TODO: Maybe we just copy the parent agent's config and override chat_model?
        #   -- but what if parent agent has a MockLMConfig?
        llm_config = lm.OpenAIGPTConfig(
            chat_model=self.model or "gpt-4.1-mini",  # Default model if not specified
        )
        config = ChatAgentConfig(
            llm=llm_config,
            system_message=f"""
                {self.system_message}
                
                When you are finished with your task, you MUST
                use the TOOL `{DoneTool.name()}` to end the task
                and return the result.                
            """,
        )

        # Create the sub-agent
        sub_agent = ChatAgent(config)

        # Enable the specified tools for the sub-agent
        # Convert tool names to actual tool classes using parent agent's tools_map
        if self.tools == ["ALL"]:
            # Enable all tools from the parent agent:
            # This is the list of all tools KNOWN (whether usable or handle-able or not)
            tool_classes = [
                agent.llm_tools_map[t]
                for t in agent.llm_tools_known
                if t in agent.llm_tools_map and t != self.request
                # Exclude the AgentTool itself!
            ]
        elif self.tools == ["NONE"]:
            # No tools enabled
            tool_classes = []
        else:
            # Enable only specified tools
            tool_classes = [
                agent.llm_tools_map[tool_name]
                for tool_name in self.tools
                if tool_name in agent.llm_tools_map
            ]

        # always enable the DoneTool to signal task completion
        sub_agent.enable_message(tool_classes + [DoneTool], use=True, handle=True)

        # Create a non-interactive task
        task = Task(sub_agent, interactive=False)

        # Run the task on the prompt, and return the result
        result = task.run(self.prompt, turns=self.max_iterations or 10)
        return result
