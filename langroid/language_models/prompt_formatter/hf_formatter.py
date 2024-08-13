"""
Prompt formatter based on HuggingFace `AutoTokenizer.apply_chat_template` method
from their Transformers library. It searches the hub for a model matching the
specified name, and uses the first one it finds. We assume that all matching
models will have the same tokenizer, so we just use the first one.
"""

import logging
import re
from typing import Any, List, Set, Tuple, Type

from jinja2.exceptions import TemplateError

from langroid.language_models.base import LanguageModel, LLMMessage, Role
from langroid.language_models.config import HFPromptFormatterConfig
from langroid.language_models.prompt_formatter.base import PromptFormatter

logger = logging.getLogger(__name__)


def try_import_hf_modules() -> Tuple[Type[Any], Type[Any]]:
    """
    Attempts to import the AutoTokenizer class from the transformers package.
    Returns:
        The AutoTokenizer class if successful.
    Raises:
        ImportError: If the transformers package is not installed.
    """
    try:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        return AutoTokenizer, HfApi
    except ImportError:
        raise ImportError(
            """
            You are trying to use some/all of:
            HuggingFace transformers.AutoTokenizer,
            huggingface_hub.HfApi,
            but these are not not installed 
            by default with Langroid. Please install langroid using the 
            `transformers` extra, like so:
            pip install "langroid[transformers]"
            or equivalent.
            """
        )


def find_hf_formatter(model_name: str) -> str:
    AutoTokenizer, HfApi = try_import_hf_modules()
    hf_api = HfApi()
    # try to find a matching model, with progressivly shorter prefixes of model_name
    model_name = model_name.lower().split("/")[-1]
    parts = re.split("[:\\-_]", model_name)
    parts = [p.lower() for p in parts if p != ""]
    for i in range(len(parts), 0, -1):
        prefix = "-".join(parts[:i])
        models = hf_api.list_models(
            task="text-generation",
            model_name=prefix,
        )
        try:
            mdl = next(models)
            tokenizer = AutoTokenizer.from_pretrained(mdl.id)
            if tokenizer.chat_template is not None:
                return str(mdl.id)
            else:
                continue
        except Exception:
            continue

    return ""


class HFFormatter(PromptFormatter):
    models: Set[str] = set()  # which models have been used for formatting

    def __init__(self, config: HFPromptFormatterConfig):
        super().__init__(config)
        AutoTokenizer, HfApi = try_import_hf_modules()
        self.config: HFPromptFormatterConfig = config
        hf_api = HfApi()
        models = hf_api.list_models(
            task="text-generation",
            model_name=config.model_name,
        )
        try:
            mdl = next(models)
        except StopIteration:
            raise ValueError(f"Model {config.model_name} not found on HuggingFace Hub")

        self.tokenizer = AutoTokenizer.from_pretrained(mdl.id)
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"Model {config.model_name} does not support chat template"
            )
        elif mdl.id not in HFFormatter.models:
            # only warn if this is the first time we've used this mdl.id
            logger.warning(
                f"""
            Using HuggingFace {mdl.id} for prompt formatting: 
            This is the CHAT TEMPLATE. If this is not what you intended,
            consider specifying a more complete model name for the formatter.
             
            {self.tokenizer.chat_template}
            """
            )
        HFFormatter.models.add(mdl.id)

    def format(self, messages: List[LLMMessage]) -> str:
        sys_msg, chat_msgs, user_msg = LanguageModel.get_chat_history_components(
            messages
        )
        # build msg dicts expected by AutoTokenizer.apply_chat_template
        sys_msg_dict = dict(role=Role.SYSTEM.value, content=sys_msg)
        chat_dicts = []
        for user, assistant in chat_msgs:
            chat_dicts.append(dict(role=Role.USER.value, content=user))
            chat_dicts.append(dict(role=Role.ASSISTANT.value, content=assistant))
        chat_dicts.append(dict(role=Role.USER.value, content=user_msg))
        all_dicts = [sys_msg_dict] + chat_dicts
        try:
            # apply chat template
            result = self.tokenizer.apply_chat_template(all_dicts, tokenize=False)
        except TemplateError:
            # this likely means the model doesn't support a system msg,
            # so combine it with the first user msg
            first_user_msg = chat_msgs[0][0] if len(chat_msgs) > 0 else user_msg
            first_user_msg = sys_msg + "\n\n" + first_user_msg
            chat_dicts[0] = dict(role=Role.USER.value, content=first_user_msg)
            result = self.tokenizer.apply_chat_template(chat_dicts, tokenize=False)
        return str(result)
