"""
Structured messages to an agent, typically from an LLM, to be handled by
an agent. The messages could represent, for example:
- information or data given to the agent
- request for information or data from the agent
- request to run a method of the agent
"""

import copy
import json
import textwrap
from abc import ABC
from random import choice
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from docstring_parser import parse

from langroid.language_models.base import LLMFunctionSpec
from langroid.pydantic_v1 import BaseModel, Extra
from langroid.utils.pydantic_utils import (
    _recursive_purge_dict_key,
    generate_simple_schema,
)
from langroid.utils.types import is_instance_of

K = TypeVar("K")


def remove_if_exists(k: K, d: dict[K, Any]) -> None:
    """Removes key `k` from `d` if present."""
    if k in d:
        d.pop(k)


def format_schema_for_strict(schema: Any) -> None:
    """
    Recursively set additionalProperties to False and replace
    oneOf and allOf with anyOf, required for OpenAI structured outputs.
    Additionally, remove all defaults and set all fields to required.
    This may not be equivalent to the original schema.
    """
    if isinstance(schema, dict):
        if "type" in schema and schema["type"] == "object":
            schema["additionalProperties"] = False

            if "properties" in schema:
                properties = schema["properties"]
                all_properties = list(properties.keys())
                for k, v in properties.items():
                    if "default" in v:
                        if k == "request":
                            v["enum"] = [v["default"]]

                        v.pop("default")
                schema["required"] = all_properties
            else:
                schema["properties"] = {}
                schema["required"] = []

        anyOf = (
            schema.get("oneOf", []) + schema.get("allOf", []) + schema.get("anyOf", [])
        )
        if "allOf" in schema or "oneOf" in schema or "anyOf" in schema:
            schema["anyOf"] = anyOf

        remove_if_exists("allOf", schema)
        remove_if_exists("oneOf", schema)

        for v in schema.values():
            format_schema_for_strict(v)
    elif isinstance(schema, list):
        for v in schema:
            format_schema_for_strict(v)


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
    """

    request: str
    purpose: str
    id: str = ""  # placeholder for OpenAI-API tool_call_id

    # If enabled, forces strict adherence to schema.
    # Currently only supported by OpenAI LLMs. When unset, enables if supported.
    _strict: Optional[bool] = None
    _allow_llm_use: bool = True  # allow an LLM to use (i.e. generate) this tool?

    # Optional param to limit number of result tokens to retain in msg history.
    # Some tools can have large results that we may not want to fully retain,
    # e.g. result of a db query, which the LLM later reduces to a summary, so
    # in subsequent dialog we may only want to retain the summary,
    # and replace this raw result truncated to _max_retained_tokens.
    # Important to note: unlike _max_result_tokens, this param is used
    # NOT used to immediately truncate the result;
    # it is only used to truncate what is retained in msg history AFTER the
    # response to this result.
    _max_retained_tokens: int | None = None

    # Optional param to limit number of tokens in the result of the tool.
    _max_result_tokens: int | None = None

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = False
        validate_all = True
        validate_assignment = True
        # do not include these fields in the generated schema
        # since we don't require the LLM to specify them
        schema_extra = {"exclude": {"purpose", "id"}}

    @classmethod
    def name(cls) -> str:
        return str(cls.default_value("request"))  # redundant str() to appease mypy

    @classmethod
    def instructions(cls) -> str:
        """
        Instructions on tool usage.
        """
        return ""

    @classmethod
    def langroid_tools_instructions(cls) -> str:
        """
        Instructions on tool usage when `use_tools == True`, i.e.
        when using langroid built-in tools
        (as opposed to OpenAI-like function calls/tools).
        """
        return """
        IMPORTANT: When using this or any other tool/function, you MUST include a 
        `request` field and set it equal to the FUNCTION/TOOL NAME you intend to use.
        """

    @classmethod
    def require_recipient(cls) -> Type["ToolMessage"]:
        class ToolMessageWithRecipient(cls):  # type: ignore
            recipient: str  # no default, so it is required

        return ToolMessageWithRecipient

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        """
        Examples to use in few-shot demos with formatting instructions.
        Each example can be either:
        - just a ToolMessage instance, e.g. MyTool(param1=1, param2="hello"), or
        - a tuple (description, ToolMessage instance), where the description is
            a natural language "thought" that leads to the tool usage,
            e.g. ("I want to find the square of 5",  SquareTool(num=5))
            In some scenarios, including such a description can significantly
            enhance reliability of tool use.
        Returns:
        """
        return []

    @classmethod
    def usage_examples(cls, random: bool = False) -> str:
        """
        Instruction to the LLM showing examples of how to use the tool-message.

        Args:
            random (bool): whether to pick a random example from the list of examples.
                Set to `true` when using this to illustrate a dialog between LLM and
                user.
                (if false, use ALL examples)
        Returns:
            str: examples of how to use the tool/function-call
        """
        # pick a random example of the fields
        if len(cls.examples()) == 0:
            return ""
        if random:
            examples = [choice(cls.examples())]
        else:
            examples = cls.examples()
        formatted_examples = [
            (
                f"EXAMPLE {i}: (THOUGHT: {ex[0]}) => \n{ex[1].format_example()}"
                if isinstance(ex, tuple)
                else f"EXAMPLE {i}:\n {ex.format_example()}"
            )
            for i, ex in enumerate(examples, 1)
        ]
        return "\n\n".join(formatted_examples)

    def to_json(self) -> str:
        return self.json(indent=4, exclude=self.Config.schema_extra["exclude"])

    def format_example(self) -> str:
        return self.json(indent=4, exclude=self.Config.schema_extra["exclude"])

    def dict_example(self) -> Dict[str, Any]:
        return self.dict(exclude=self.Config.schema_extra["exclude"])

    def get_value_of_type(self, target_type: Type[Any]) -> Any:
        """Try to find a value of a desired type in the fields of the ToolMessage."""
        ignore_fields = self.Config.schema_extra["exclude"].union(["request"])
        for field_name in set(self.dict().keys()) - ignore_fields:
            value = getattr(self, field_name)
            if is_instance_of(value, target_type):
                return value
        return None

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
    def format_instructions(cls, tool: bool = False) -> str:
        """
        Default Instructions to the LLM showing how to use the tool/function-call.
        Works for GPT4 but override this for weaker LLMs if needed.

        Args:
            tool: instructions for Langroid-native tool use? (e.g. for non-OpenAI LLM)
                (or else it would be for OpenAI Function calls).
                Ignored in the default implementation, but can be used in subclasses.
        Returns:
            str: instructions on how to use the message
        """
        # TODO: when we attempt to use a "simpler schema"
        # (i.e. all nested fields explicit without definitions),
        # we seem to get worse results, so we turn it off for now
        param_dict = (
            # cls.simple_schema() if tool else
            cls.llm_function_schema(request=True).parameters
        )
        examples_str = ""
        if cls.examples():
            examples_str = "EXAMPLES:\n" + cls.usage_examples()
        return textwrap.dedent(
            f"""
            TOOL: {cls.default_value("request")}
            PURPOSE: {cls.default_value("purpose")} 
            JSON FORMAT: {
                json.dumps(param_dict, indent=4)
            }
            {examples_str}
            """.lstrip()
        )

    @staticmethod
    def group_format_instructions() -> str:
        """Template for instructions for a group of tools.
        Works with GPT4 but override this for weaker LLMs if needed.
        """
        return textwrap.dedent(
            """
            === ALL AVAILABLE TOOLS and THEIR FORMAT INSTRUCTIONS ===
            You have access to the following TOOLS to accomplish your task:

            {format_instructions}
            
            When one of the above TOOLs is applicable, you must express your 
            request as "TOOL:" followed by the request in the above format.
            """
        )

    @classmethod
    def llm_function_schema(
        cls,
        request: bool = False,
        defaults: bool = True,
    ) -> LLMFunctionSpec:
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
            defaults: whether to include fields with default values in the schema,
                    in the "properties" section.

        Returns:
            LLMFunctionSpec: the schema as an LLMFunctionSpec

        """
        schema = copy.deepcopy(cls.schema())
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

        excludes = cls.Config.schema_extra["exclude"]
        if not request:
            excludes = excludes.union({"request"})
        # exclude 'excludes' from parameters["properties"]:
        parameters["properties"] = {
            field: details
            for field, details in parameters["properties"].items()
            if field not in excludes and (defaults or details.get("default") is None)
        }
        parameters["required"] = sorted(
            k
            for k, v in parameters["properties"].items()
            if ("default" not in v and k not in excludes)
        )
        if request:
            parameters["required"].append("request")

            # If request is present it must match the default value
            # Similar to defining request as a literal type
            parameters["request"] = {
                "enum": [cls.default_value("request")],
                "type": "string",
            }

        if "description" not in schema:
            if docstring.short_description:
                schema["description"] = docstring.short_description
            else:
                schema["description"] = (
                    f"Correctly extracted `{cls.__name__}` with all "
                    f"the required parameters with correct types"
                )

        # Handle nested ToolMessage fields
        if "definitions" in parameters:
            for v in parameters["definitions"].values():
                if "exclude" in v:
                    v.pop("exclude")

                    remove_if_exists("purpose", v["properties"])
                    remove_if_exists("id", v["properties"])
                    if (
                        "request" in v["properties"]
                        and "default" in v["properties"]["request"]
                    ):
                        if "required" not in v:
                            v["required"] = []
                        v["required"].append("request")
                        v["properties"]["request"] = {
                            "type": "string",
                            "enum": [v["properties"]["request"]["default"]],
                        }

        parameters.pop("exclude")
        _recursive_purge_dict_key(parameters, "title")
        _recursive_purge_dict_key(parameters, "additionalProperties")
        return LLMFunctionSpec(
            name=cls.default_value("request"),
            description=cls.default_value("purpose"),
            parameters=parameters,
        )

    @classmethod
    def simple_schema(cls) -> Dict[str, Any]:
        """
        Return a simplified schema for the message, with only the request and
        required fields.
        Returns:
            Dict[str, Any]: simplified schema
        """
        schema = generate_simple_schema(
            cls,
            exclude=list(cls.Config.schema_extra["exclude"]),
        )
        return schema
