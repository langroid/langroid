import logging
from typing import List, Tuple

from langroid.language_models.base import LanguageModel, LLMMessage
from langroid.language_models.config import Llama2FormatterConfig
from langroid.language_models.prompt_formatter.base import PromptFormatter

logger = logging.getLogger(__name__)


BOS: str = "<s>"
EOS: str = "</s>"
B_INST: str = "[INST]"
E_INST: str = "[/INST]"
B_SYS: str = "<<SYS>>\n"
E_SYS: str = "\n<</SYS>>\n\n"
SPECIAL_TAGS: List[str] = [B_INST, E_INST, BOS, EOS, "<<SYS>>", "<</SYS>>"]


class Llama2Formatter(PromptFormatter):
    def __int__(self, config: Llama2FormatterConfig) -> None:
        super().__init__(config)
        self.config: Llama2FormatterConfig = config

    def format(self, messages: List[LLMMessage]) -> str:
        sys_msg, chat_msgs, user_msg = LanguageModel.get_chat_history_components(
            messages
        )
        return self._get_prompt_from_components(sys_msg, chat_msgs, user_msg)

    def _get_prompt_from_components(
        self,
        system_prompt: str,
        chat_history: List[Tuple[str, str]],
        user_message: str,
    ) -> str:
        """
        For llama2 models, convert chat history into a single
        prompt for Llama2 models, for use in the /completions endpoint
        (as opposed to the /chat/completions endpoint).
        See:
        https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/
        https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L44

        Args:
            system_prompt (str): system prompt, typically specifying role/task.
            chat_history (List[Tuple[str,str]]): List of (user, assistant) pairs
            user_message (str): user message, at the end of the chat, i.e. the message
                for which we want to generate a response.

        Returns:
            str: Prompt for Llama2 models

        Typical structure of the formatted prompt:
        Note important that the first [INST], [/INST] surrounds the system prompt,
        together with the first user message. A lot of libs seem to miss this detail.

        <s>[INST] <<SYS>>
        You are are a helpful... bla bla.. assistant
        <</SYS>>

        Hi there! [/INST] Hello! How can I help you today? </s><s>[INST]
        What is a neutron star? [/INST] A neutron star is a ... </s><s>
        [INST] Okay cool, thank you! [/INST] You're welcome! </s><s>
        [INST] Ah, I have one more question.. [/INST]
        """
        bos = BOS if self.config.use_bos_eos else ""
        eos = EOS if self.config.use_bos_eos else ""
        text = f"{bos}{B_INST} {B_SYS}{system_prompt}{E_SYS}"
        for user_input, response in chat_history:
            text += (
                f"{user_input.strip()} {E_INST} {response.strip()} {eos}{bos} {B_INST} "
            )
        text += f"{user_message.strip()} {E_INST}"
        return text
