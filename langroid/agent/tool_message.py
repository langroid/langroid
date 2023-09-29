"""
Structured messages to an agent, typically from an LLM, to be handled by
an agent. The messages could represent, for example:
- information or data given to the agent
- request for information or data from the agent
- request to run a method of the agent
"""

from abc import ABC
from random import choice
from typing import Any, Dict, List, Type

from docstring_parser import parse
from pydantic import BaseModel

from langroid.language_models.base import LLMFunctionSpec


def _recursive_purge_dict_key(d: Dict[str, Any], k: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == k and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_dict_key(d[key], k)


class ToolMessage(ABC, BaseModel):
    """
    Abstract Class for a class that defines the structure of a "Tool" message from an
    LLM. Depending on context, "tools" are also referred to as "plugins",
    or "function calls" (in the context of OpenAI LLMs).
    Essentially, they are a way for the LLM to express its intent to run a special
    function or method. Currently these "tools" are handled by methods of the
    agent.

    Attributes:
        request (str): name of agent method to map to.
        purpose (str): purpose of agent method, expressed in general terms.
            (This is used when auto-generating the tool instruction to the LLM)
        result (str): example of result of agent method.
    """

    request: str
    purpose: str
    result: str = ""
    recipient: str = ""  # default is empty string, so it is optional

    class Config:
        arbitrary_types_allowed = False
        validate_all = True
        validate_assignment = True
        # do not include these fields in the generated schema
        # since we don't require the LLM to specify them
        schema_extra = {"exclude": {"purpose", "result"}}

    @classmethod
    def instructions(cls) -> str:
        return ""

    @classmethod
    def require_recipient(cls) -> Type["ToolMessage"]:
        class ToolMessageWithRecipient(cls):  # type: ignore
            recipient: str  # no default, so it is required

        return ToolMessageWithRecipient

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        """
        Examples to use in few-shot demos with JSON formatting instructions.
        Returns:
        """
        return []

    @classmethod
    def usage_example(cls) -> str:
        """
        Instruction to the LLM showing an example of how to use the message.
        Returns:
            str: example of how to use the message
        """
        # pick a random example of the fields
        if len(cls.examples()) == 0:
            return ""
        ex = choice(cls.examples())
        return ex.json_example()

    def json_example(self) -> str:
        return self.json(indent=4, exclude={"result", "purpose"})

    def dict_example(self) -> Dict[str, Any]:
        return self.dict(exclude={"result", "purpose"})

    @classmethod
    def default_value(cls, f: str) -> Any:
        """
        Returns the default value of the given field, for the message-class
        Args:
            f (str): field name

        Returns:
            Any: default value of the field, or None if not set or if the
                field does not exist.
        """
        schema = cls.schema()
        properties = schema["properties"]
        return properties.get(f, {}).get("default", None)

    @classmethod
    def llm_function_schema(cls, request: bool = False) -> LLMFunctionSpec:
        """
        Clean up the schema of the Pydantic class (which can recursively contain
        other Pydantic classes), to create a version compatible with OpenAI
        Function-call API.

        Adapted from this excellent library:
        https://github.com/jxnl/instructor/blob/main/instructor/function_calls.py

        Args:
            request: whether to include the "request" field in the schema.
                (we set this to True when using Langroid-native TOOLs as opposed to
                OpenAI Function calls)

        Returns:
            LLMFunctionSpec: the schema as an LLMFunctionSpec

        """
        schema = cls.schema()
        docstring = parse(cls.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        excludes = (
            ["result", "purpose"] if request else ["request", "result", "purpose"]
        )
        # exclude 'excludes' from parameters["properties"]:
        parameters["properties"] = {
            field: details
            for field, details in parameters["properties"].items()
            if field not in excludes
        }
        parameters["required"] = sorted(
            k
            for k, v in parameters["properties"].items()
            if ("default" not in v and k not in excludes)
        )
        if request:
            parameters["required"].append("request")

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        parameters.pop("exclude")
        _recursive_purge_dict_key(parameters, "title")
        _recursive_purge_dict_key(parameters, "additionalProperties")
        return LLMFunctionSpec(
            name=cls.default_value("request"),
            description=cls.default_value("purpose"),
            parameters=parameters,
        )
