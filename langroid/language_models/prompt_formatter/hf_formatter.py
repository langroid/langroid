"""
Prompt formatter based on HuggingFace `AutoTokenizer.apply_chat_template` method
from their Transformers library. It searches the hub for models matching the
specified name, most-downloaded first, and uses the first one whose tokenizer
loads and has a chat template. We assume that all matching models share the
same chat template.
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


# How many hub matches to try before giving up on a name.
MAX_HUB_CANDIDATES = 10


def _hub_candidates(hf_api: Any, model_name: str) -> List[str]:
    """Hub model ids to try for `model_name`, best first.

    A full `org/name` id is tried as given first. The rest are hub search
    hits, most-downloaded first: the hub's default order puts arbitrary
    repackagings (e.g. ONNX exports with no loadable tokenizer) ahead of the
    canonical model.

    Args:
        hf_api: A `huggingface_hub.HfApi` instance.
        model_name: The model name or id to search for.

    Returns:
        Candidate model ids, without duplicates.
    """
    ids = [model_name] if "/" in model_name else []
    for mdl in hf_api.list_models(
        task="text-generation",
        model_name=model_name,
        sort="downloads",
        limit=MAX_HUB_CANDIDATES,
    ):
        if mdl.id not in ids:
            ids.append(mdl.id)
    return ids


def _load_chat_tokenizer(
    AutoTokenizer: Type[Any], ids: List[str]
) -> Tuple[str, Any] | None:
    """Return the first `(id, tokenizer)` that loads and has a chat template.

    Args:
        AutoTokenizer: The `transformers.AutoTokenizer` class.
        ids: Candidate hub model ids, in order of preference.

    Returns:
        The model id and its tokenizer, or None if no candidate qualifies.
    """
    for mdl_id in ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(mdl_id)
        except Exception as e:
            logger.info(f"Skipping {mdl_id} for prompt formatting: {e}")
            continue
        if tokenizer.chat_template is not None:
            return mdl_id, tokenizer
    return None


def find_hf_formatter(model_name: str) -> str:
    AutoTokenizer, HfApi = try_import_hf_modules()
    hf_api = HfApi()
    # try to find a matching model, with progressivly shorter prefixes of model_name
    model_name = model_name.lower().split("/")[-1]
    parts = re.split("[:\\-_]", model_name)
    parts = [p.lower() for p in parts if p != ""]
    for i in range(len(parts), 0, -1):
        prefix = "-".join(parts[:i])
        try:
            found = _load_chat_tokenizer(AutoTokenizer, _hub_candidates(hf_api, prefix))
        except Exception:
            continue
        if found is not None:
            return found[0]

    return ""


class HFFormatter(PromptFormatter):
    models: Set[str] = set()  # which models have been used for formatting

    def __init__(self, config: HFPromptFormatterConfig):
        super().__init__(config)
        AutoTokenizer, HfApi = try_import_hf_modules()
        self.config: HFPromptFormatterConfig = config
        hf_api = HfApi()
        ids = _hub_candidates(hf_api, config.model_name)
        if not ids:
            raise ValueError(f"Model {config.model_name} not found on HuggingFace Hub")
        found = _load_chat_tokenizer(AutoTokenizer, ids)
        if found is None:
            raise ValueError(
                f"Model {config.model_name} does not support chat template"
            )
        mdl_id, self.tokenizer = found
        if mdl_id not in HFFormatter.models:
            # only warn if this is the first time we've used this mdl.id
            logger.warning(
                f"""
            Using HuggingFace {mdl_id} for prompt formatting: 
            This is the CHAT TEMPLATE. If this is not what you intended,
            consider specifying a more complete model name for the formatter.
             
            {self.tokenizer.chat_template}
            """
            )
        HFFormatter.models.add(mdl_id)

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
