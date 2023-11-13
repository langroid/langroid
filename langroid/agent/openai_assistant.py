import asyncio

# setup logger
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Type, cast, no_type_check

from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run, ThreadMessage
from pydantic import BaseModel
from rich import print

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.base import LLMFunctionCall, LLMMessage, LLMResponse, Role
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from langroid.utils.configuration import settings
from langroid.utils.system import generate_user_id, update_hash

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    RETRIEVAL = "retrieval"
    CODE_INTERPRETER = "code_interpreter"
    FUNCTION = "function"


class AssitantTool(BaseModel):
    type: ToolType
    function: Dict[str, Any] | None = None

    def dct(self) -> Dict[str, Any]:
        d = super().dict()
        d["type"] = d["type"].value
        return d


class AssistantToolCall(BaseModel):
    id: str
    type: ToolType
    function: LLMFunctionCall


class RunStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_ACTION = "requires_action"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class OpenAIAssistantConfig(ChatAgentConfig):
    use_cached_assistant: bool = False  # set in script via user dialog
    assistant_id: str | None = None
    use_cached_thread: bool = False  # set in script via user dialog
    thread_id: str | None = None
    # set to True once we can add Assistant msgs in threads
    cache_responses: bool = False
    timeout: int = 30  # can be different from llm.timeout
    llm = OpenAIGPTConfig()
    tools: List[AssitantTool] = []
    files: List[str] = []


class OpenAIAssistant(ChatAgent):

    """
    A ChatAgent powered by OpenAI Assistant API:
    mainly, in `llm_response` method, we avoid maintaining conversation state,
    and instead let the Assistant API do it for us.
    Also handles persistent storage of Assistant and Threads:
    stores their ids (for given user, org) in a cache, and
    reuses them based on config.use_cached_assistant and config.use_cached_thread.

    This class can be used as a drop-in replacement for ChatAgent.
    """

    def __init__(self, config: OpenAIAssistantConfig):
        super().__init__(config)
        self.config: OpenAIAssistantConfig = config
        self.llm: OpenAIGPT = OpenAIGPT(self.config.llm)
        # handles for various entities and methods
        self.client = self.llm.client
        self.runs = self.llm.client.beta.threads.runs
        self.threads = self.llm.client.beta.threads
        self.thread_messages = self.llm.client.beta.threads.messages
        self.assistants = self.llm.client.beta.assistants
        # which tool_ids are awaiting output submissions
        self.pending_tool_ids: List[str] = []

        self.thread: Thread | None = None
        self.assistant: Assistant | None = None
        self.run: Run | None = None

        self._maybe_create_assistant(self.config.assistant_id)
        self._maybe_create_thread(self.config.thread_id)
        self._cache_store()

        self.add_assistant_files(self.config.files)
        self.add_assistant_tools(self.config.tools)
        # TODO remove this once OpenAI supports storing Assistant msgs in threads
        settings.cache = False

    def add_assistant_files(self, files: List[str]) -> None:
        """Add file_ids to assistant"""
        if self.assistant is None:
            raise ValueError("Assistant is None")
        self.files = [
            self.client.files.create(file=open(f, "rb"), purpose="assistants")
            for f in files
        ]
        self.config.files = list(set(self.config.files + files))
        self.assistant = self.assistants.update(
            self.assistant.id,
            file_ids=[f.id for f in self.files],
        )

    def add_assistant_tools(self, tools: List[AssitantTool]) -> None:
        """Add tools to assistant"""
        if self.assistant is None:
            raise ValueError("Assistant is None")
        all_tool_dicts = [t.dct() for t in self.config.tools]
        for t in tools:
            if t.dct() not in all_tool_dicts:
                self.config.tools.append(t)
        self.assistant = self.assistants.update(
            self.assistant.id,
            tools=[tool.dct() for tool in self.config.tools],  # type: ignore
        )

    def enable_message(
        self,
        message_class: Optional[Type[ToolMessage]],
        use: bool = True,
        handle: bool = True,
        force: bool = False,
        require_recipient: bool = False,
    ) -> None:
        """Override ChatAgent's method: extract the function-related args"""
        super().enable_message(
            message_class,
            use=use,
            handle=handle,
            force=force,
            require_recipient=require_recipient,
        )
        if message_class is None or not use:
            # no specific msg class, or
            # we are not enabling USAGE/GENERATION of this tool/fn,
            # then there's no need to attach the fn to the assistant
            # (HANDLING the fn will still work via self.agent_response)
            return
        if self.config.use_tools:
            sys_msg = self._create_system_and_tools_message()
            self.set_system_message(sys_msg.content)
        if not self.config.use_functions_api:
            return
        functions, _ = self._function_args()
        if functions is None:
            return
        # add the functions to the assistant:
        if self.assistant is None:
            raise ValueError("Assistant is None")
        tools = self.assistant.tools
        tools.extend(
            [
                {
                    "type": "function",  # type: ignore
                    "function": f.dict(),
                }
                for f in functions
            ]
        )
        self.assistant = self.assistants.update(
            self.assistant.id,
            tools=tools,  # type: ignore
        )

    def _cache_thread_key(self) -> str:
        """Key to use for caching or retrieving thread id"""
        org = self.llm.client.organization or ""
        uid = generate_user_id(org)
        name = self.config.name
        return "Thread:" + name + ":" + uid

    def _cache_assistant_key(self) -> str:
        """Key to use for caching or retrieving assistant id"""
        org = self.llm.client.organization or ""
        uid = generate_user_id(org)
        name = self.config.name
        return "Assistant:" + name + ":" + uid

    @no_type_check
    def _cache_messages_key(self) -> str:
        """Key to use when caching or retrieving thread messages"""
        if self.thread is None:
            raise ValueError("Thread is None")
        return "Messages:" + self.thread.metadata["hash"]

    @no_type_check
    def _cache_thread_lookup(self) -> str | None:
        """Try to retrieve cached thread_id associated with
        this user + machine + organization"""
        key = self._cache_thread_key()
        return self.llm.cache.retrieve(key)

    @no_type_check
    def _cache_assistant_lookup(self) -> str | None:
        """Try to retrieve cached assistant_id associated with
        this user + machine + organization"""
        key = self._cache_assistant_key()
        return self.llm.cache.retrieve(key)

    @no_type_check
    def _cache_messages_lookup(self) -> str | None:
        """Try to retrieve cached response for the message-list-hash"""
        if not settings.cache:
            return None
        key = self._cache_messages_key()
        return self.llm.cache.retrieve(key)

    def _cache_store(self) -> None:
        """
        Cache the assistant_id, thread_id associated with
        this user + machine + organization
        """
        if self.thread is None or self.assistant is None:
            raise ValueError("Thread or Assistant is None")
        thread_key = self._cache_thread_key()
        self.llm.cache.store(thread_key, self.thread.id)

        assistant_key = self._cache_assistant_key()
        self.llm.cache.store(assistant_key, self.assistant.id)

    @staticmethod
    def thread_msg_to_llm_msg(msg: ThreadMessage) -> LLMMessage:
        """
        Convert a ThreadMessage to an LLMMessage
        """
        return LLMMessage(
            content=msg.content[0].text.value,  # type: ignore
            role=msg.role,
        )

    def _update_messages_hash(self, msg: ThreadMessage | LLMMessage) -> None:
        """
        Update the hash of messages in the thread with the latest message
        """
        if self.thread is None:
            raise ValueError("Thread is None")
        if isinstance(msg, ThreadMessage):
            llm_msg = self.thread_msg_to_llm_msg(msg)
        else:
            llm_msg = msg
        hash = self.thread.metadata["hash"]  # type: ignore
        most_recent_msg = llm_msg.content
        most_recent_role = llm_msg.role
        hash = update_hash(hash, f"{most_recent_role}:{most_recent_msg}")
        # TODO is this inplace?
        self.thread = self.threads.update(
            self.thread.id,
            metadata={
                "hash": hash,
            },
        )
        assert self.thread.metadata["hash"] == hash  # type: ignore

    def _maybe_create_thread(self, id: str | None = None) -> None:
        """Retrieve or create a thread if one does not exist,
        or retrieve it from cache"""
        if id is not None:
            try:
                self.thread = self.threads.retrieve(thread_id=id)
            except Exception:
                logger.warning(
                    f"""
                    Could not retrieve thread with id {id}, 
                    so creating a new one.
                    """
                )
                self.thread = None
            if self.thread is not None:
                return
        cached = self._cache_thread_lookup()
        if cached is not None:
            if self.config.use_cached_thread:
                self.thread = self.llm.client.beta.threads.retrieve(thread_id=cached)
            else:
                logger.warning(
                    f"""
                    Found cached thread id {cached}, 
                    but config.use_cached_thread = False, so deleting it.
                    """
                )
                self.llm.client.beta.threads.delete(thread_id=cached)
                self.llm.cache.delete_keys([self._cache_thread_key()])
        if self.thread is None:
            if self.assistant is None:
                raise ValueError("Assistant is None")
            self.thread = self.llm.client.beta.threads.create()
            hash_hex = update_hash(
                None,
                s=self.assistant.instructions or "",
            )
            self.thread = self.threads.update(
                self.thread.id,
                metadata={
                    "hash": hash_hex,
                },
            )
            assert self.thread.metadata["hash"] == hash_hex  # type: ignore

    def _maybe_create_assistant(self, id: str | None = None) -> None:
        """Retrieve or create an assistant if one does not exist,
        or retrieve it from cache"""
        if id is not None:
            try:
                self.assistant = self.assistants.retrieve(assistant_id=id)
            except Exception:
                logger.warning(
                    f"""
                    Could not retrieve assistant with id {id}, 
                    so creating a new one.
                    """
                )
                self.assistant = None
            if self.assistant is not None:
                return
        cached = self._cache_assistant_lookup()
        if cached is not None:
            if self.config.use_cached_assistant:
                self.assistant = self.llm.client.beta.assistants.retrieve(
                    assistant_id=cached
                )
            else:
                logger.warning(
                    f"""
                    Found cached assistant id {cached}, 
                    but config.use_cached_assistant = False, so deleting it.
                    """
                )
                self.llm.client.beta.assistants.delete(assistant_id=cached)
                self.llm.cache.delete_keys([self._cache_assistant_key()])
        if self.assistant is None:
            self.assistant = self.llm.client.beta.assistants.create(
                name=self.config.name,
                instructions=self.config.system_message,
                tools=[],
                model=self.config.llm.chat_model,
            )

    def _get_run(self) -> Run:
        """Retrieve the run object associated with this thread and run,
        to see its latest status.
        """
        if self.thread is None or self.run is None:
            raise ValueError("Thread or Run is None")
        return self.runs.retrieve(thread_id=self.thread.id, run_id=self.run.id)

    def _add_thread_message(self, msg: str, role: Role) -> None:
        """
        Add a message with the given role to the thread.
        Args:
            msg (str): message to add
            role (Role): role of the message
        """
        if self.thread is None:
            raise ValueError("Thread is None")
        thread_msg = self.thread_messages.create(
            content=msg,
            thread_id=self.thread.id,
            role=role,  # type: ignore
        )
        self._update_messages_hash(thread_msg)

    def _get_thread_messages(self, n: int = 20) -> List[LLMMessage]:
        """
        Get the last n messages in the thread, in cleaned-up form (LLMMessage).
        Args:
            n (int): number of messages to retrieve
        Returns:
            List[LLMMessage]: list of messages
        """
        if self.thread is None:
            raise ValueError("Thread is None")
        result = self.thread_messages.list(
            thread_id=self.thread.id,
            limit=n,
        )
        num = len(result.data)
        if result.has_more and num < n:  # type: ignore
            logger.warning(f"Retrieving last {num} messages, but there are more")
        thread_msgs = result.data
        for msg in thread_msgs:
            self.process_citations(msg)
        return [
            LLMMessage(
                # TODO: could be image, deal with it later
                content=m.content[0].text.value,  # type: ignore
                role=m.role,
            )
            for m in thread_msgs
        ]

    def _wait_for_run(
        self,
        until_not: List[RunStatus] = [RunStatus.QUEUED, RunStatus.IN_PROGRESS],
        until: List[RunStatus] = [],
        timeout: int = 30,
    ) -> RunStatus:
        """
        Poll the run until it either:
        - EXITs the statuses specified in `until_not`, or
        - ENTERs the statuses specified in `until`, or
        """
        if self.thread is None or self.run is None:
            raise ValueError("Thread or Run is None")
        while True:
            run = self._get_run()
            if run.status not in until_not or run.status in until:
                return cast(RunStatus, run.status)
            time.sleep(1)
            timeout -= 1
            if timeout <= 0:
                return cast(RunStatus, RunStatus.TIMEOUT)

    async def _wait_for_run_async(
        self,
        until_not: List[RunStatus] = [RunStatus.QUEUED, RunStatus.IN_PROGRESS],
        until: List[RunStatus] = [],
        timeout: int = 30,
    ) -> RunStatus:
        """Async version of _wait_for_run"""
        if self.thread is None or self.run is None:
            raise ValueError("Thread or Run is None")
        while True:
            run = self._get_run()
            if run.status not in until_not or run.status in until:
                return cast(RunStatus, run.status)
            await asyncio.sleep(1)
            timeout -= 1
            if timeout <= 0:
                return cast(RunStatus, RunStatus.TIMEOUT)

    def set_system_message(self, msg: str) -> None:
        """
        Override ChatAgent's method.
        The Task may use this method to set the system message
        of the chat assistant.
        """
        super().set_system_message(msg)
        if self.assistant is None:
            raise ValueError("Assistant is None")
        self.assistant = self.assistants.update(self.assistant.id, instructions=msg)

    def _start_run(self) -> None:
        """
        Run the assistant on the thread.
        """
        if self.thread is None or self.assistant is None:
            raise ValueError("Thread or Assistant is None")
        self.run = self.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )

    def _run_result(self) -> LLMResponse:
        """Result from run completed on the thread."""
        status = self._wait_for_run(
            timeout=self.config.timeout,
        )
        return self._process_run_result(status)

    async def _run_result_async(self) -> LLMResponse:
        """(Async) Result from run completed on the thread."""
        status = await self._wait_for_run_async(
            timeout=self.config.timeout,
        )
        return self._process_run_result(status)

    def _process_run_result(self, status: RunStatus) -> LLMResponse:
        """Process the result of the run."""
        function_call: LLMFunctionCall | None = None
        response = ""
        tool_id = ""
        if status == RunStatus.TIMEOUT:
            logger.warning("Timeout waiting for run to complete, return empty string")
        elif status == RunStatus.COMPLETED:
            messages = self._get_thread_messages(n=1)
            response = messages[0].content
            # IMPORTANT: FIRST get hash key to store result,
            # THEN update hash, since this will now include the response!
            key = self._cache_messages_key()
            self._update_messages_hash(messages[0])
            self.llm.cache.store(key, response)
        elif status == RunStatus.REQUIRES_ACTION:
            tool_calls = self._parse_run_required_action()
            # pick the FIRST tool call with type "function"
            tool_call_fn = [t for t in tool_calls if t.type == ToolType.FUNCTION][0]
            # TODO Handling only first tool/fn call for now
            # revisit later: multi-tools affects the task.run() loop.
            function_call = tool_call_fn.function
            tool_id = tool_call_fn.id
        return LLMResponse(
            message=response,
            tool_id=tool_id,
            function_call=function_call,
            usage=None,  # TODO
            cached=False,  # TODO - revisit when able to insert Assistant responses
        )

    def _parse_run_required_action(self) -> List[AssistantToolCall]:
        """
        Parse the required_action field of the run, i.e. get the list of tool calls.
        Currently only tool calls are supported.
        """
        # see https://platform.openai.com/docs/assistants/tools/function-calling
        run = self._get_run()
        if run.status != RunStatus.REQUIRES_ACTION:  # type: ignore
            return []

        if (action := run.required_action.type) != "submit_tool_outputs":
            raise ValueError(f"Unexpected required_action type {action}")
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        return [
            AssistantToolCall(
                id=tool_call.id,
                type=ToolType(tool_call.type),
                function=LLMFunctionCall.from_dict(tool_call.function.model_dump()),
            )
            for tool_call in tool_calls
        ]

    def _submit_tool_outputs(self, msg: LLMMessage) -> None:
        """
        Submit the tool (fn) outputs to the run/thread
        """
        if self.run is None or self.thread is None:
            raise ValueError("Run or Thread is None")
        tool_outputs = [
            {
                "tool_call_id": msg.tool_id,
                "output": msg.content,
            }
        ]
        # run enters queued, in_progress state after this
        self.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs,  # type: ignore
        )

    def process_citations(self, thread_msg: ThreadMessage) -> None:
        """
        Process citations in the thread message.
        Modifies the thread message in-place.
        """
        # could there be multiple content items?
        # TODO content could be MessageContentImageFile; handle that later
        annotated_content = thread_msg.content[0].text  # type: ignore
        annotations = annotated_content.annotations
        citations = []
        # Iterate over the annotations and add footnotes
        for index, annotation in enumerate(annotations):
            # Replace the text with a footnote
            annotated_content.value = annotated_content.value.replace(
                annotation.text, f" [{index}]"
            )
            # Gather citations based on annotation attributes
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = self.client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] '{file_citation.quote}',-- from {cited_file.filename}"
                )
            elif file_path := getattr(annotation, "file_path", None):
                cited_file = self.client.files.retrieve(file_path.file_id)
                citations.append(
                    f"[{index}] Click <here> to download {cited_file.filename}"
                )
            # Note: File download functionality not implemented above for brevity
        sep = "\n" if len(citations) > 0 else ""
        annotated_content.value += sep + "\n".join(citations)

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """
        Override ChatAgent's method: this is the main LLM response method.
        In the ChatAgent, this updates `self.message_history` and then calls
        `self.llm_response_messages`, but since we are relying on the Assistant API
        to maintain conversation state, this method is simpler: Simply start a run
        on the message-thread, and wait for it to complete.

        Args:
            message (Optional[str | ChatDocument], optional): message to respond to
                (if absent, the LLM response will be based on the
                instructions in the system_message). Defaults to None.
        Returns:
            Optional[ChatDocument]: LLM response
        """
        is_tool_output = False
        if message is not None:
            llm_msg = ChatDocument.to_LLMMessage(message)
            tool_id = llm_msg.tool_id
            if tool_id in self.pending_tool_ids:
                if isinstance(message, ChatDocument):
                    message.pop_tool_ids()
                is_tool_output = True
                # submit tool/fn result to the thread/run
                self._submit_tool_outputs(llm_msg)
                self.pending_tool_ids.remove(tool_id)
            else:
                # add message to the thread
                self._add_thread_message(llm_msg.content, role=Role.USER)

        # When message is None, the thread may have no user msgs,
        # Note: system message is NOT placed in the thread by the OpenAI system.

        # check if we have cached the response.
        # TODO: handle the case of structured result (fn-call, tool, etc)
        result = self._cache_messages_lookup()
        cached = False
        if result is not None:
            cached = True
            # store the result in the thread so
            # it looks like assistant produced it
            # TODO Adding an Assistant msg is currently NOT supported by the API,
            # so we cannot use it until it is.
            if self.config.cache_responses:
                self._add_thread_message(result, role=Role.ASSISTANT)
        else:
            # create a run for this assistant on this thread,
            # i.e. actually "run"
            if not is_tool_output:
                # DO NOT start a run if we submitted tool outputs,
                # since submission of tool outputs resumes a run from
                # status = "requires_action"
                self._start_run()
            response = self._run_result()

        # code from ChatAgent.llm_response_messages
        if response.function_call is not None:
            self.pending_tool_ids += [response.tool_id]
            response_str = str(response.function_call)
        else:
            response_str = response.message
        cache_str = "[red](cached)[/red]" if cached else ""
        if not settings.quiet:
            print(f"{cache_str}[green]" + response_str + "[/green]")
        cdoc = ChatDocument.from_LLMResponse(response, displayed=False)
        # Note message.metadata.tool_ids may have been popped above
        tool_ids = (
            []
            if (message is None or isinstance(message, str))
            else message.metadata.tool_ids
        )

        if response.tool_id != "":
            tool_ids.append(response.tool_id)
        cdoc.metadata.tool_ids = tool_ids
        return cdoc

    async def llm_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """
        Override ChatAgent's method: this is the main LLM response method.
        In the ChatAgent, this updates `self.message_history` and then calls
        `self.llm_response_messages`, but since we are relying on the Assistant API
        to maintain conversation state, this method is simpler: Simply start a run
        on the message-thread, and wait for it to complete.

        Args:
            message (Optional[str | ChatDocument], optional): message to respond to
                (if absent, the LLM response will be based on the
                instructions in the system_message). Defaults to None.
        Returns:
            Optional[ChatDocument]: LLM response
        """
        is_tool_output = False
        if message is not None:
            llm_msg = ChatDocument.to_LLMMessage(message)
            tool_id = llm_msg.tool_id
            if tool_id in self.pending_tool_ids:
                if isinstance(message, ChatDocument):
                    message.pop_tool_ids()
                is_tool_output = True
                # submit tool/fn result to the thread/run
                self._submit_tool_outputs(llm_msg)
                self.pending_tool_ids.remove(tool_id)
            else:
                # add message to the thread
                self._add_thread_message(llm_msg.content, role=Role.USER)

        # When message is None, the thread may have no user msgs,
        # Note: system message is NOT placed in the thread by the OpenAI system.

        # check if we have cached the response.
        # TODO: handle the case of structured result (fn-call, tool, etc)
        result = self._cache_messages_lookup()
        cached = False
        if result is not None:
            cached = True
            # store the result in the thread so
            # it looks like assistant produced it
            # TODO Adding an Assistant msg is currently NOT supported by the API,
            # so we cannot use it until it is.
            if self.config.cache_responses:
                self._add_thread_message(result, role=Role.ASSISTANT)
        else:
            # create a run for this assistant on this thread,
            # i.e. actually "run"
            if not is_tool_output:
                # DO NOT start a run if we submitted tool outputs,
                # since submission of tool outputs resumes a run from
                # status = "requires_action"
                self._start_run()
            response = await self._run_result_async()

        # code from ChatAgent.llm_response_messages
        if response.function_call is not None:
            self.pending_tool_ids += [response.tool_id]
            response_str = str(response.function_call)
        else:
            response_str = response.message
        cache_str = "[red](cached)[/red]" if cached else ""
        if not settings.quiet:
            print(f"{cache_str}[green]" + response_str + "[/green]")
        cdoc = ChatDocument.from_LLMResponse(response, displayed=False)
        # Note message.metadata.tool_ids may have been popped above
        tool_ids = (
            []
            if (message is None or isinstance(message, str))
            else message.metadata.tool_ids
        )

        if response.tool_id != "":
            tool_ids.append(response.tool_id)
        cdoc.metadata.tool_ids = tool_ids
        return cdoc
