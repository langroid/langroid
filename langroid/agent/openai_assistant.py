import asyncio

# setup logger
import logging
import time
from typing import List, Optional, no_type_check

from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run, ThreadMessage
from rich import print

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global, settings
from langroid.utils.system import generate_user_id, update_hash

logger = logging.getLogger(__name__)


class OpenAIAssistantConfig(ChatAgentConfig):
    use_cached_assistant: bool = True  # set in script via user dialog
    use_cached_thread: bool = True  # set in script via user dialog
    # set to True once we can add Assistant msgs in threads
    cache_responses: bool = False
    timeout: int = 30  # can be different from llm.timeout
    llm = OpenAIGPTConfig()


class OpenAIAssistant(ChatAgent):

    """
    A ChatAgent powered by OpenAI Assistant API:
    mainly, in `llm_response` method, we avoid maintaining conversation state,
    and instead let the Assistant API do it for us.
    Also handles persistent storage of Assistant and Threads:
    stores their ids (for given user, org) in a cache, and
    reuses them based on config.use_cached_assistant and config.use_cached_thread.
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

        self.thread: Thread | None = None
        self.assistant: Assistant | None = None
        self.run: Run | None = None

        self._maybe_create_assistant()
        self._maybe_create_thread()
        self._cache_store()

        # TODO remove this once OpenAI supports storing Assistant msgs in threads
        set_global(Settings(cache=False))

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

    def _update_messages_hash(self) -> None:
        """
        Update the hash of messages in the thread with the latest message
        """
        if self.thread is None:
            raise ValueError("Thread is None")
        hash = self.thread.metadata["hash"]  # type: ignore
        most_recent_msg = self._get_thread_messages(n=1)[0].content
        most_recent_role = self._get_thread_messages(n=1)[0].role
        hash = update_hash(hash, f"{most_recent_role}:{most_recent_msg}")
        # TODO is this inplace?
        self.thread = self.threads.update(
            self.thread.id,
            metadata={
                "hash": hash,
            },
        )
        assert self.thread.metadata["hash"] == hash  # type: ignore

    def _maybe_create_thread(self) -> None:
        """Create a thread if one does not exist, or retrieve it from cache"""
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

    def _maybe_create_assistant(self) -> None:
        """Create an assistant if one does not exist, or retrieve it from cache"""
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
        self.thread_messages.create(
            content=msg,
            thread_id=self.thread.id,
            role=role,  # type: ignore
        )
        self._update_messages_hash()

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

    def _wait_for_run_status(
        self, status: str = "completed", timeout: int = 30
    ) -> bool:
        """Poll the run status until it has specified status, or timeout."""
        if self.thread is None or self.run is None:
            raise ValueError("Thread or Run is None")
        while True:
            run = self.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id,
            )
            if run.status == status:
                return True
            time.sleep(1)
            timeout -= 1
            if timeout <= 0:
                return False

    async def _wait_for_run_status_async(
        self, status: str = "completed", timeout: int = 30
    ) -> bool:
        """Async Poll the run status until it has specified status, or timeout."""
        if self.thread is None or self.run is None:
            raise ValueError("Thread or Run is None")
        while True:
            run = self.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id,
            )
            if run.status == status:
                return True
            await asyncio.sleep(1)
            timeout -= 1
            if timeout <= 0:
                return False

    def set_system_message(self, msg: str) -> None:
        """
        Override ChatAgent's method.
        The Task may use this method to set the system message
        of the chat assistant.
        """
        super().set_system_message(msg)
        if self.assistant is None:
            raise ValueError("Assistant is None")
        # TODO this is an inplace update: revisit this line if it causes problems
        self.assistants.update(self.assistant.id, instructions=msg)

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

    def _run_result(self) -> str:
        """Result from run completed on the thread."""
        done = self._wait_for_run_status(
            status="completed",
            timeout=self.config.timeout,
        )
        return self._process_run_result(done)

    async def _run_result_async(self) -> str:
        """(Async) Result from run completed on the thread."""
        done = await self._wait_for_run_status_async(
            status="completed",
            timeout=self.config.timeout,
        )
        return self._process_run_result(done)

    def _process_run_result(self, done: bool) -> str:
        """Process the result of the run."""
        if not done:
            logger.warning("Timeout waiting for run to complete, return empty string")
            result = ""
        else:
            messages = self._get_thread_messages(n=1)
            result = messages[0].content
            # IMPORTANT: FIRST get hash key to store result,
            # THEN update hash, since this will now include the response!
            key = self._cache_messages_key()
            self._update_messages_hash()
            self.llm.cache.store(key, result)
        return result

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
                    f"[{index}] {file_citation.quote} from {cited_file.filename}"
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
        if message is not None:
            message_str = (
                message.content if (isinstance(message, ChatDocument)) else message
            )
            # add message to the thread
            self._add_thread_message(message_str, role=Role.USER)

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
            self._start_run()
            result = self._run_result()
        cache_str = "[red](cached)[/red]" if cached else ""
        print(f"{cache_str}[green]" + result + "[/green]")
        return ChatDocument(
            content=result,
            metadata=ChatDocMetaData(
                source=Entity.LLM,
                sender=Entity.LLM,
                sender_name=self.config.name,
            ),
        )

    async def llm_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """Async version of llm_response"""

        if message is not None:
            message_str = (
                message.content if (isinstance(message, ChatDocument)) else message
            )
            # add message to the thread
            self._add_thread_message(message_str, role=Role.USER)

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
            self._start_run()
            result = await self._run_result_async()
        cache_str = "[red](cached)[/red]" if cached else ""
        print(f"{cache_str}[green]" + result + "[/green]")
        return ChatDocument(
            content=result,
            metadata=ChatDocMetaData(
                source=Entity.LLM,
                sender=Entity.LLM,
                sender_name=self.config.name,
            ),
        )
