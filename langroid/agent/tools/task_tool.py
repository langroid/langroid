"""
TaskTool: A tool that allows agents to delegate a task to a sub-agent with
    specific tools enabled.
"""

import uuid
from typing import List, Optional

from pydantic import Field
from pydantic.fields import ModelPrivateAttr

import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from langroid.mytypes import Entity


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
    model: Optional[str] = Field(
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

    def _user_origin(self, chat_doc: Optional[ChatDocument] = None) -> bool:
        """Did this `task_tool` request derive from raw user input?

        Three signals, the same ones `Agent._filter_user_origin_tools` reads,
        so that "raw user input" means here exactly what it means there:

        - this tool object was parsed out of tainted (USER-derived) content;
        - the carrying document is tainted, i.e. external user input or a
          mechanical derivation of it (`ChatDocument.from_str` and
          `to_ChatDocument` set this for `Entity.USER`);
        - the carrying document was authored by `Entity.USER` and its tools
          did NOT come from an agent. `ChatDocMetaData.tainted` defaults to
          `False`, so an application that builds its own USER `ChatDocument`
          rather than going through those two constructors would otherwise
          look LLM-origin here. The `tools_from_agent` exemption keeps
          legitimate agent-to-agent handoffs (a Task relabels the relayed
          sender to USER) on the LLM-origin path.

        With NO carrying document the origin cannot be established, so this
        fails closed and answers "user" -- the same reading the framework's
        filter gives a USER-sender document, which is what an origin-less
        message most often is. `Agent.handle_tool_message` omits `chat_doc`
        when the handled message was a bare string (`has_chat_doc_arg`
        requires `chat_doc is not None`); it is not worth guessing an LLM
        behind such a message, and a caller that genuinely means "LLM-origin"
        can say so by calling :meth:`_set_up_task` directly.

        For the record, since it is easy to misread the dispatcher: the two
        `handle_tool_message(tool)` calls in `ChatAgent`'s strict-recovery
        path (`chat_agent.py`) do NOT reach this method with `chat_doc=None`.
        The generated handler wrapper declares `chat_doc` without a default
        (`Agent._analyze_handler_params`), so that call raises `TypeError`
        before any handler body runs -- behavior this fix does not change.
        """
        if self._tainted:
            return True
        if chat_doc is None:
            return True
        return bool(
            chat_doc.metadata.tainted
            or (
                chat_doc.metadata.sender == Entity.USER
                and not chat_doc.metadata.tools_from_agent
            )
        )

    def _sub_agent_tools(
        self, agent: ChatAgent, user_origin: bool
    ) -> tuple[List[type[ToolMessage]], List[str]]:
        """Resolve the tool classes to enable in the sub-agent.

        A USER-origin `task_tool` request may only hand the sub-agent tools
        that the parent agent lets a user invoke, i.e. the tools the parent
        enabled with `use=True` (`agent.llm_tools_usable`). Without that
        restriction one LLM-usable tool (TaskTool) lets raw user text name any
        tool merely REGISTERED in the parent -- including the documented
        sub-agent-only pattern `enable_message(T, use=False, handle=False)` --
        and run its handler with arbitrary arguments, since the sub-agent
        enables the named tools for use AND handling and the user's own prompt
        text reaches that handler before the sub-agent's model is called
        (GHSA-78x2-mgm6-3qgc).

        An LLM-origin request keeps today's behavior: the parent's LLM decided
        to delegate, so the sub-agent-only pattern must keep working, and only
        the tool CLASS's `_allow_llm_use` applies.

        This method only reads the parent's tool registry, so it is safe to
        call more than once per request.

        Args:
            agent: The parent ChatAgent that is handling this tool
            user_origin: whether this request derived from raw user input

        Returns:
            `(tool_classes, refused)`: the classes to enable in the sub-agent,
            and the names this request asked for by name but may not have.
        """

        def llm_usable_class(tool_class: type[ToolMessage]) -> bool:
            allow_llm_use = tool_class._allow_llm_use
            if isinstance(allow_llm_use, ModelPrivateAttr):
                allow_llm_use = allow_llm_use.default
            return bool(allow_llm_use)

        def permitted(tool_name: str, tool_class: type[ToolMessage]) -> bool:
            if user_origin:
                # the parent's PER-AGENT policy decides, not just the class
                return tool_name in agent.llm_tools_usable
            return llm_usable_class(tool_class)

        tool_classes: List[type[ToolMessage]] = []
        refused: List[str] = []
        if self.tools == ["ALL"]:
            # Enable all tools from the parent agent:
            # This is the list of all tools KNOWN (whether usable or handle-able or not)
            for t in agent.llm_tools_known:
                if t in agent.llm_tools_map and t != self.request:
                    tool_class = agent.llm_tools_map[t]
                    if permitted(t, tool_class):
                        tool_classes.append(tool_class)
        elif self.tools == ["NONE"]:
            # No tools enabled
            pass
        else:
            # Enable only specified tools
            for tool_name in self.tools:
                if tool_name not in agent.llm_tools_map:
                    continue
                tool_class = agent.llm_tools_map[tool_name]
                if permitted(tool_name, tool_class):
                    tool_classes.append(tool_class)
                elif user_origin:
                    # Asked for a registered tool the parent does not let a
                    # user invoke: refuse the request in writing rather than
                    # silently running a sub-task without the tool.
                    refused.append(tool_name)
        return tool_classes, refused

    def _refusal_message(self, refused: List[str]) -> str:
        """Message returned instead of running the sub-task, when a USER-origin
        request named tools the parent does not let a user invoke.

        Every interpolated value is framework-controlled: `refused` holds names
        that were found in the parent's `llm_tools_map`, and the request name
        comes from the CLASS default (`default_value`) -- which is also what
        dispatch matches on -- rather than from this instance's `request`
        field, which a caller can set to anything. A refusal must not echo
        attacker-chosen text into a document another agent may read.
        """
        names = ", ".join(sorted(refused))
        return f"""
        Error: this `{self.default_value("request")}` request came from user
        input, so the sub-agent may only be given tools that a user is allowed
        to invoke in this agent. These requested tools are NOT among them, and
        no sub-task was run: {names}.
        """

    def _set_up_task(self, agent: ChatAgent, *, user_origin: bool) -> Task:
        """
        Helper method to set up a task for the sub-agent.

        Args:
            agent: The parent ChatAgent that is handling this tool
            user_origin: whether this request derived from raw user input; such
                a request gets only the parent's user-invocable tools
                (see :meth:`_sub_agent_tools`). Required and keyword-only ON
                PURPOSE: it decides a security policy, so a caller that forgot
                it must fail loudly rather than silently get the permissive
                reading.
        """
        # Generate a random name if not provided
        agent_name = self.agent_name or f"agent-{str(uuid.uuid4())[:8]}"

        # Create chat agent config with system message if provided
        # `not self.model` rather than `is None`: the field is optional and an
        # LLM routinely fills an optional string with `""` instead of omitting
        # it, which would take the fallback branch below and hand a Claude or
        # local-endpoint parent an OpenAI sub-agent -- the very bug this
        # inheritance exists to fix, through a different door. The fallback
        # already reads `""` as "no model given" (`self.model or ...`).
        if not self.model and agent.config.llm is not None:
            # Inherit the parent's config so the sub-agent reaches the same
            # provider and endpoint, but carry every CALLABLE field over by
            # identity rather than deep-copying it, for the two reasons
            # `OpenAIGPT.__init__` gives where it does the same dance for
            # `api_key_provider`: a callable may hold non-copyable state (a
            # `threading.Lock` in a refreshing credential provider, say),
            # which makes `deepcopy` raise outright; and where the identity is
            # load-bearing, a copy silently breaks it -- the client cache is
            # keyed on `id(api_key_provider)`, so a copied provider gives
            # every sub-agent its own client.
            #
            # Every callable field is excluded rather than a named list,
            # because the named list was already wrong once: `streamer`,
            # `streamer_async`, `run_on_first_use`, `http_client_factory` and
            # `MockLMConfig.response_fn`/`response_fn_async` are all callables
            # on configs reachable here. Doing it by identity costs nothing,
            # since `deepcopy` of a plain function or class returns the very
            # same object anyway -- only stateful callable OBJECTS behave
            # differently, and those are exactly the ones to preserve.
            parent_llm_config = agent.config.llm
            callable_fields = {
                name: value
                for name in type(parent_llm_config).model_fields
                if callable(value := getattr(parent_llm_config, name, None))
            }
            if callable_fields:
                parent_llm_config = parent_llm_config.model_copy(
                    update=dict.fromkeys(callable_fields)
                )
            llm_config = parent_llm_config.model_copy(deep=True)
            for name, value in callable_fields.items():
                # `setattr` rather than attribute syntax because `llm_config`
                # is typed as the base `LLMConfig`, which declares only some
                # of these fields.
                setattr(llm_config, name, value)
        else:
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
        tool_classes, _ = self._sub_agent_tools(agent, user_origin)

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
        user_origin = self._user_origin(chat_doc)
        refused = self._sub_agent_tools(agent, user_origin)[1]
        if refused:
            # tainted: the framework taints an agent response to tainted
            # input (#1035); this one bypasses `response_template`, so it
            # must carry the mark itself -- a refusal only happens on the
            # user-origin path.
            return agent.create_agent_response(
                content=self._refusal_message(refused), tainted=True
            )

        task = self._set_up_task(agent, user_origin=user_origin)

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
                    # the sub-task seed derives from chat_doc AND from
                    # this tool's own fields (#1035)
                    tainted=chat_doc.metadata.tainted or self._tainted,
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
        user_origin = self._user_origin(chat_doc)
        refused = self._sub_agent_tools(agent, user_origin)[1]
        if refused:
            # tainted: the framework taints an agent response to tainted
            # input (#1035); this one bypasses `response_template`, so it
            # must carry the mark itself -- a refusal only happens on the
            # user-origin path.
            return agent.create_agent_response(
                content=self._refusal_message(refused), tainted=True
            )

        task = self._set_up_task(agent, user_origin=user_origin)

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
                    # the sub-task seed derives from chat_doc AND from
                    # this tool's own fields (#1035)
                    tainted=chat_doc.metadata.tainted or self._tainted,
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
