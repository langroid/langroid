"""
Structured messages to an agent, typically from an LLM, to be handled by
an agent. The messages could represent, for example:
- information or data given to the agent
- request for information or data from the agent
- request to run a method of the agent
"""

import json
import textwrap
from abc import ABC
from random import choice
from typing import Any, Dict, List, Tuple, Type

from docstring_parser import parse

from langroid.language_models.base import LLMFunctionSpec
from langroid.pydantic_v1 import BaseModel, Extra
from langroid.utils.pydantic_utils import (
    _recursive_purge_dict_key,
    generate_simple_schema,
)
from langroid.utils.types import is_instance_of


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

    _allow_llm_use: bool = True  # allow an LLM to use (i.e. generate) this tool?

    # model_config = ConfigDict(extra=Extra.allow)

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = False
        validate_all = True
        validate_assignment = True
        # do not include these fields in the generated schema
        # since we don't require the LLM to specify them
        schema_extra = {"exclude": {"purpose", "id"}}

    @classmethod
    def instructions(cls) -> str:
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
        Examples to use in few-shot demos with JSON formatting instructions.
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
        examples_jsons = [
            (
                f"EXAMPLE {i}: (THOUGHT: {ex[0]}) => \n{ex[1].json_example()}"
                if isinstance(ex, tuple)
                else f"EXAMPLE {i}:\n {ex.json_example()}"
            )
            for i, ex in enumerate(examples, 1)
        ]
        return "\n\n".join(examples_jsons)

    def to_json(self) -> str:
        return self.json(indent=4, exclude=self.Config.schema_extra["exclude"])

    def json_example(self) -> str:
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
    def json_instructions(cls, tool: bool = False) -> str:
        """
        Default Instructions to the LLM showing how to use the tool/function-call.
        Works for GPT4 but override this for weaker LLMs if needed.

        Args:
            tool: instructions for Langroid-native tool use? (e.g. for non-OpenAI LLM)
                (or else it would be for OpenAI Function calls)
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
    def json_group_instructions() -> str:
        """Template for instructions for a group of tools.
        Works with GPT4 but override this for weaker LLMs if needed.
        """
        return textwrap.dedent(
            """
            === ALL AVAILABLE TOOLS and THEIR JSON FORMAT INSTRUCTIONS ===
            You have access to the following TOOLS to accomplish your task:

            {json_instructions}
            
            When one of the above TOOLs is applicable, you must express your 
            request as "TOOL:" followed by the request in the above JSON format.
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
