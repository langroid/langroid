from typing import Optional, List, Dict, Callable

from pydantic.v1.networks import MultiHostDsn

from langroid.agent.chat_document import ChatDocMetaData
from langroid.agent.tool_message import format_schema_for_strict
from sglang import (
    function,
    system,
    user,
    assistant,
    gen,
    set_default_backend,
    OpenAI,
    RuntimeEndpoint,
)
import json
from sglang.lang.interpreter import ProgramState
from outlines.fsm.json_schema import build_regex_from_schema
import langroid as lr
import langroid.language_models as lm
from dotenv import load_dotenv
from rich import print
from langroid import ChatDocument
from langroid.language_models import LLMMessage
from langroid.language_models.base import LLMResponse, ToolChoiceTypes
from langroid import Entity

load_dotenv()

# config = lm.AzureConfig()
# set_default_backend(OpenAI(
#     config.model_name,
#     is_azure=True,
#     api_key=config.api_key,
#     azure_endpoint=config.api_base,
#     api_version=config.api_version,
#     azure_deployment=config.deployment_name,
# ))
set_default_backend(OpenAI("gpt-4o"))

def default_response_fn(s: ProgramState) -> None:
    s += assistant(gen("answer", max_tokens=100))

class SGLAgentConfig(lr.ChatAgentConfig):
    response_function: Callable[[ProgramState], None] = default_response_fn

class SGLAgent(lr.ChatAgent):
    def __init__(self, config: SGLAgentConfig = SGLAgentConfig()):
        super().__init__(config)
        self.response_function = config.response_function

    # Handling message history updates here, TODO in llm_response
    def llm_response_messages(
        self,
        messages: List[LLMMessage],
        output_len: Optional[int] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
    ) -> ChatDocument:
        # Skip first generation step
        # TODO fix
        if len(messages) == 1:
            message = "Please give me your question."
            print(message)
            return ChatDocument.from_LLMResponse(LLMResponse(
                message=message,
            ))

        @function
        def multi_turn_question(s):
            # initialize program state with message history
            for m in messages:
                role = m.role
                content = m.content
                role_dict = dict(assistant=assistant, user=user, system=system)
                s += role_dict[role](content)

            self.response_function(s)

        state = multi_turn_question.run(stream=True)

        # Ignoring first len(messages) messages, appending additional messages
        # to history
        role_dict = dict(user=lm.Role.USER, assistant=lm.Role.ASSISTANT, system=lm.Role.SYSTEM)
        for message in state.messages()[len(messages):]:
            msg = lm.LLMMessage(
                content=message["content"],
                role=role_dict[message["role"]],
            )
            self.message_history.append(msg)

        entity_dict = {
            lm.Role.USER: Entity.USER,
            lm.Role.ASSISTANT: Entity.LLM,
            lm.Role.SYSTEM: Entity.SYSTEM,
        }
        msg = self.message_history[-1]
        # Hack: remove last message so `llm_response` appends
        # TODO fix
        self.message_history = self.message_history[:-1]
        print(state.messages())
        
        msg = lr.ChatDocument(
            content=msg.content,
            metadata=ChatDocMetaData(
                source=entity_dict[msg.role],
                sender=entity_dict[msg.role],
            ),
        )
        return msg

agent = SGLAgent()
lr.Task(agent).run(turns=3)

# We can express multiple generation steps
def multi_step_fn(s: ProgramState) -> None:
    s += assistant(gen("answer", max_tokens=100))
    s += user("Now, explain why you gave that answer")
    s += assistant(gen("explanation", max_tokens=200))

agent_multi_step = SGLAgent(
    SGLAgentConfig(
        response_function=multi_step_fn,
    )
)
lr.Task(agent_multi_step, single_round=True).run(turns=3)


# Only works locally
set_default_backend(RuntimeEndpoint("http://localhost:30000"))
class MultiplicationTool(lr.ToolMessage):
    request: str = "muliply"
    purpose: str = "To multiply two numbers"
    x: int
    y: int

    def handle(self) -> str:
        return str(self.x * self.y)

class AdditionTool(lr.ToolMessage):
    request: str = "add"
    purpose: str = "To add two numbers"
    x: int
    y: int

    def handle(self) -> str:
        return f"DONE {str(self.x + self.y)}"

def to_regex(tool: type[lr.ToolMessage]) -> str:
    schema = tool.llm_function_schema(request=True).parameters
    # format_schema_for_strict(schema)
    return build_regex_from_schema(json.dumps(schema))

# We can express tool calls during the generation step
def function_calling_fn(s: ProgramState) -> None:
    s += assistant(gen("multiplication", regex=to_regex(MultiplicationTool)))
    # Handle during generation
    total = MultiplicationTool.parse_raw(s["multiplication"]).handle()
    s += user(f"The product is {total}. Now, add 3 to that value with the `add` tool.")
    # Handle via usual Langroid approach
    s += assistant(gen("addition", regex=to_regex(AdditionTool)))

agent_tool = SGLAgent(
    SGLAgentConfig(
        response_function=function_calling_fn,
    )
)
agent_tool.enable_message([MultiplicationTool, AdditionTool])
lr.Task(agent_tool, interactive=False).run("What is  15 * 11? Use the `multiplication` tool.")
