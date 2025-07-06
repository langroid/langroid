"""
TaskTool: A tool that allows agents to delegate a task to a sub-agent with
    specific tools enabled.
"""

import uuid
from typing import List, Optional

import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.pydantic_v1 import Field


class TaskTool(ToolMessage):
    """
    Tool that spawns a sub-agent with specified tools to handle a task.

    The sub-agent can be given a custom name for identification in logs.
    If no name is provided, a random unique name starting with 'agent'
    will be generated.
    """

    # TODO: setting up termination conditions of sub-task needs to be improved
    request: str = "task_tool"
    purpose: str = """
        <HowToUse>
        Use this tool to delegate a task to a sub-agent with specific tools enabled.
        The sub-agent will be created with the specified tools and will run the task
        non-interactively.
    """

    # Parameters for the agent tool

    system_message: Optional[str] = Field(
        ...,
        description="""
        Optional system message to configure the sub-agent's general behavior and 
        to specify the task and its context.
            A good system message will have these components:
            - Inform the sub-agent of its role, e.g. "You are a financial analyst."
            - Clear spec of the task, with sufficient context for the sub-agent to 
              understand what it needs to do, since the sub-agent does 
              NOT have access to your conversation history!
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
            GENERAL role and goals, the `prompt` is the SPECIFIC input that the 
            sub-agent will process. In LLM terms, the system message is sent to the 
            LLM as the first message, with role = "system" or "developer", and 
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
        If you want to enable all tools, or you do not have any preference
        on what tools are enabled for the sub-agent, you can set 
        this field to a singleton list ['ALL']
        To disable all tools, set it to a singleton list ['NONE']
        """,
    )
    # TODO: ensure valid model name
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
    agent_name: Optional[str] = Field(
        default=None,
        description="""
            Optional name for the sub-agent. This will be used as the agent's name
            in logs and for identification purposes. If not provided, a random unique
            name starting with 'agent' will be generated.
            """,
    )

    def _set_up_task(self, agent: ChatAgent) -> Task:
        """
        Helper method to set up a task for the sub-agent.

        Args:
            agent: The parent ChatAgent that is handling this tool
        """
        # Generate a random name if not provided
        agent_name = self.agent_name or f"agent-{str(uuid.uuid4())[:8]}"

        # Create chat agent config with system message if provided
        # TODO: Maybe we just copy the parent agent's config and override chat_model?
        #   -- but what if parent agent has a MockLMConfig?
        llm_config = lm.OpenAIGPTConfig(
            chat_model=self.model or lm.OpenAIChatModel.GPT4_1_MINI,
        )
        config = ChatAgentConfig(
            name=agent_name,
            llm=llm_config,
            handle_llm_no_tool=f"""
                You forgot to use one of your TOOLs! Remember that you must either:
                - use a tool, or a sequence of tools, to complete your task, OR
                - if you are done with your task, use the `{DoneTool.name()}` tool
                to return the result.
                
                As a reminder, this was your task:
                {self.prompt}
                """,
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
                if t in agent.llm_tools_map
                and t != self.request
                and agent.llm_tools_map[t]._allow_llm_use
                # Exclude the TaskTool itself!
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
                and agent.llm_tools_map[tool_name]._allow_llm_use
            ]

        # always enable the DoneTool to signal task completion
        sub_agent.enable_message(tool_classes + [DoneTool], use=True, handle=True)

        # Create a non-interactive task
        task = Task(sub_agent, interactive=False)

        return task

    def handle(
        self, agent: ChatAgent, chat_doc: Optional[ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """

        Handle the TaskTool by creating a sub-agent with specified tools
        and running the task non-interactively.

        Args:
            agent: The parent ChatAgent that is handling this tool
            chat_doc: The ChatDocument containing this tool message
        """

        task = self._set_up_task(agent)

        # Create a ChatDocument for the prompt with parent pointer
        prompt_doc = None
        if chat_doc is not None:
            from langroid.agent.chat_document import ChatDocMetaData

            prompt_doc = ChatDocument(
                content=self.prompt,
                metadata=ChatDocMetaData(
                    parent_id=chat_doc.id(),
                    agent_id=agent.id,
                    sender=chat_doc.metadata.sender,
                ),
            )
            # Set bidirectional parent-child relationship
            chat_doc.metadata.child_id = prompt_doc.id()

        # Run the task with the ChatDocument or string prompt
        result = task.run(prompt_doc or self.prompt, turns=self.max_iterations or 10)
        return result

    async def handle_async(
        self, agent: ChatAgent, chat_doc: Optional[ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """
        Async method to handle the TaskTool by creating a sub-agent with specified tools
        and running the task non-interactively.

        Args:
            agent: The parent ChatAgent that is handling this tool
            chat_doc: The ChatDocument containing this tool message
        """
        task = self._set_up_task(agent)

        # Create a ChatDocument for the prompt with parent pointer
        prompt_doc = None
        if chat_doc is not None:
            from langroid.agent.chat_document import ChatDocMetaData

            prompt_doc = ChatDocument(
                content=self.prompt,
                metadata=ChatDocMetaData(
                    parent_id=chat_doc.id(),
                    agent_id=agent.id,
                    sender=chat_doc.metadata.sender,
                ),
            )
            # Set bidirectional parent-child relationship
            chat_doc.metadata.child_id = prompt_doc.id()

        # Run the task with the ChatDocument or string prompt
        # TODO eventually allow the various task setup configs,
        #  including termination conditions
        result = await task.run_async(
            prompt_doc or self.prompt, turns=self.max_iterations or 10
        )
        return result
