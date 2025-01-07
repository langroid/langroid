from typing import Optional, List, Dict

from sglang import (
    function,
    system,
    user,
    assistant,
    gen,
    set_default_backend,
    OpenAI,
)
import langroid as lr
import langroid.language_models as lm
from dotenv import load_dotenv
from rich import print
from langroid import ChatDocument
from langroid.language_models import LLMMessage
from langroid.language_models.base import ToolChoiceTypes

load_dotenv()

set_default_backend(OpenAI("gpt-4o"))

class SGLAgent(lr.ChatAgent):
    def llm_response_messages(
        self,
        messages: List[LLMMessage],
        output_len: Optional[int] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
    ) -> ChatDocument:

        @function
        def multi_turn_question(s):
            for m in messages:
                role = m.role
                content = m.content
                role_dict = dict(assistant=assistant, user=user, system=system)
                s += role_dict[role](content)
            s += assistant(gen("answer", max_tokens=output_len))
            # TODO - handle tool_choice

        state = multi_turn_question.run(stream=True)
        response = lm.LLMResponse(message=state["answer"])
        # TODO bunch of other fields to fill in
        return lr.ChatDocument.from_LLMResponse(response)


agent = SGLAgent()

while True:
    user_input = input("User: ")
    if user_input in ["q", "x"]:
        break
    print("[green]Assistant: ", agent.llm_response(user_input).content)
