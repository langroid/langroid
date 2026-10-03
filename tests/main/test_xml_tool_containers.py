"""Tests that XML tool-calls preserve the declared types of collection fields.

`XMLToolMessage.extract_field_values` infers structure from the XML alone, which
loses type information in two ways: an empty element looks like an empty string
rather than an empty list/dict, and a single-entry dict (or single-field nested
model) looks like a list because all its children share a tag. Both make
otherwise valid tool-calls fail Pydantic validation.
"""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.language_models.mock_lm import MockLMConfig


class Filters(BaseModel):
    category: str


class SearchXML(XMLToolMessage):
    request: str = "search_xml"
    purpose: str = "Search with collection arguments"
    filters: Dict[str, str]
    tags: List[str]
    options: Filters

    def handle(self) -> str:
        category = self.filters.get("category", "all")
        return f"{category}|{len(self.tags)}|{self.options.category}"


class OptionalSearchXML(XMLToolMessage):
    """Same collection fields, but declared as `Optional[...]`.

    `format_instructions` reduces `Optional[X]` to `X` when telling the LLM how
    to format a field, so parsing has to do the same reduction.
    """

    request: str = "optional_search_xml"
    purpose: str = "Search with optional collection arguments"
    filters: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None


@pytest.mark.parametrize("filters", [{}, {"category": "RAG"}, {"a": "A", "b": "B"}])
@pytest.mark.parametrize("tags", [[], ["AI"], ["AI", "RAG"]])
def test_xml_tool_collection_roundtrip(
    filters: Dict[str, str], tags: List[str]
) -> None:
    original = SearchXML(filters=filters, tags=tags, options=Filters(category="RAG"))
    parsed = SearchXML.parse(original.format_example())
    assert parsed == original


@pytest.mark.parametrize("filters", [{}, {"category": "RAG"}, {"a": "A", "b": "B"}])
@pytest.mark.parametrize("tags", [[], ["AI"], ["AI", "RAG"]])
def test_optional_xml_tool_collection_roundtrip(
    filters: Dict[str, str], tags: List[str]
) -> None:
    original = OptionalSearchXML(filters=filters, tags=tags)
    parsed = OptionalSearchXML.parse(original.format_example())
    assert parsed == original


def test_explicit_empty_collections_are_retained() -> None:
    values = SearchXML.extract_field_values(
        "<tool><request>search_xml</request><filters/><tags/>"
        "<options><category>RAG</category></options><_internal>ignore</_internal></tool>"
    )
    assert values is not None
    assert values["filters"] == {}
    assert values["tags"] == []
    assert "_internal" not in values


def test_optional_empty_collections_use_declared_types() -> None:
    values = OptionalSearchXML.extract_field_values(
        "<tool><request>optional_search_xml</request><filters/><tags/></tool>"
    )
    assert values is not None
    assert values["filters"] == {}
    assert values["tags"] == []


def test_optional_single_entry_dict_stays_a_mapping() -> None:
    """A one-key dict must not be mistaken for a one-element list."""
    values = OptionalSearchXML.extract_field_values(
        "<tool><request>optional_search_xml</request>"
        "<filters><category>RAG</category></filters></tool>"
    )
    assert values is not None
    assert values["filters"] == {"category": "RAG"}


class BareCollectionsXML(XMLToolMessage):
    """Bare `list`/`dict` annotations, which `format_instructions` also
    advertises as collections."""

    request: str = "bare_collections_xml"
    purpose: str = "Search with bare collection annotations"
    tags: list = []
    filters: dict = {}


class NestedListXML(XMLToolMessage):
    request: str = "nested_list_xml"
    purpose: str = "A list of single-field nested models"
    items: List[Filters] = []


def test_bare_collection_annotations_use_declared_types() -> None:
    values = BareCollectionsXML.extract_field_values(
        "<tool><request>bare_collections_xml</request><tags/><filters/></tool>"
    )
    assert values is not None
    assert values["tags"] == []
    assert values["filters"] == {}


def test_single_field_models_inside_a_list_stay_mappings() -> None:
    """Each list item is a one-field model, so it must not collapse to a list."""
    parsed = NestedListXML.parse(
        "<tool><request>nested_list_xml</request>"
        "<items><item><category>RAG</category></item>"
        "<item><category>XML</category></item></items></tool>"
    )
    assert parsed is not None
    assert parsed.items == [Filters(category="RAG"), Filters(category="XML")]


def test_one_single_field_model_inside_a_list() -> None:
    parsed = NestedListXML.parse(
        "<tool><request>nested_list_xml</request>"
        "<items><item><category>RAG</category></item></items></tool>"
    )
    assert parsed is not None
    assert parsed.items == [Filters(category="RAG")]


class CollidingTagsXML(XMLToolMessage):
    """A nested tag that collides with another top-level field's name.

    `<values><item>a</item></values>` must not pick up the type of the
    unrelated top-level `item` field.
    """

    request: str = "colliding_tags_xml"
    purpose: str = "Nested tag colliding with a top-level field name"
    values: List[str] = []
    item: Dict[str, str] = {}


class ModelCollisionXML(XMLToolMessage):
    """A dict key colliding with a top-level *model*-typed field's name."""

    request: str = "model_collision_xml"
    purpose: str = "Dict key colliding with a model-typed field name"
    data: Dict[str, str] = {}
    options: Filters = Filters(category="none")


class DeepModelCollisionXML(XMLToolMessage):
    """A nested *branch* element colliding with a model-typed field's name.

    Unlike `ModelCollisionXML`, the colliding element has children, so it
    reaches the nested-model construction branch.
    """

    request: str = "deep_model_collision_xml"
    purpose: str = "Nested branch colliding with a model-typed field name"
    data: Dict[str, Dict[str, str]] = {}
    options: Filters = Filters(category="none")


def test_nested_branch_colliding_with_model_typed_field() -> None:
    """`<data><options><category>x</category></options></data>` is a plain dict.

    It must not be constructed as the unrelated top-level `options: Filters`.
    """
    parsed = DeepModelCollisionXML.parse(
        "<tool><request>deep_model_collision_xml</request>"
        "<data><options><category>x</category></options></data>"
        "<options><category>RAG</category></options></tool>"
    )
    assert parsed is not None
    assert parsed.data == {"options": {"category": "x"}}
    assert parsed.options == Filters(category="RAG")


class Values(RootModel[List[str]]):
    pass


class RootModelXML(XMLToolMessage):
    """A `RootModel` wrapping a list: its children are items, not fields."""

    request: str = "root_model_xml"
    purpose: str = "RootModel-backed collection field"
    values: Values = Values(root=[])


def test_root_model_backed_list_field() -> None:
    parsed = RootModelXML.parse(
        "<tool><request>root_model_xml</request>"
        "<values><item>a</item><item>b</item></values></tool>"
    )
    assert parsed is not None
    assert parsed.values == Values(root=["a", "b"])


class VerbatimCollisionXML(XMLToolMessage):
    """A nested tag colliding with a top-level *verbatim* field's name."""

    request: str = "verbatim_collision_xml"
    purpose: str = "Nested tag colliding with a verbatim field name"
    code: str = Field("", json_schema_extra={"verbatim": True})
    payload: Dict[str, List[str]] = {}


def test_nested_tag_colliding_with_verbatim_field() -> None:
    """A nested `<code>` must not inherit the top-level `code` verbatim flag.

    Doing so parses the nested branch as raw text, losing its children.
    """
    values = VerbatimCollisionXML.extract_field_values(
        "<tool><request>verbatim_collision_xml</request>"
        "<code>x</code>"
        "<payload><code><item>a</item></code><other><item>b</item></other></payload>"
        "</tool>"
    )
    assert values is not None
    assert values["payload"] == {"code": ["a"], "other": ["b"]}
    assert values["code"] == "x"


class PrefixedPayload(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def _require_prefix(cls, v: str) -> str:
        if not v.startswith("ok:"):
            raise ValueError("value must start with 'ok:'")
        return v


class BeforeValidatorXML(XMLToolMessage):
    """A tool whose own `mode="before"` validator fixes up a nested value."""

    request: str = "before_validator_xml"
    purpose: str = "Tool-level before-validator over a nested model"
    payload: PrefixedPayload

    @model_validator(mode="before")
    @classmethod
    def _add_prefix(cls, data: Any) -> Any:
        if isinstance(data, dict):
            payload = data.get("payload")
            if isinstance(payload, dict) and "value" in payload:
                value = payload["value"]
                if isinstance(value, str) and not value.startswith("ok:"):
                    data = {**data, "payload": {**payload, "value": f"ok:{value}"}}
        return data


def test_nested_model_left_for_containing_tool_validators() -> None:
    """A nested model must not be built before the tool's own validators run.

    Building it here would validate `value="raw"` and reject the call, even
    though the tool's `mode="before"` validator would have made it valid.
    """
    parsed = BeforeValidatorXML.parse(
        "<tool><request>before_validator_xml</request>"
        "<payload><value>raw</value></payload></tool>"
    )
    assert parsed is not None
    assert parsed.payload == PrefixedPayload(value="ok:raw")


def test_nested_tag_colliding_with_top_level_field() -> None:
    parsed = CollidingTagsXML.parse(
        "<tool><request>colliding_tags_xml</request>"
        "<values><item>alpha</item></values>"
        "<item><key>value</key><other>two</other></item></tool>"
    )
    assert parsed is not None
    assert parsed.values == ["alpha"]
    assert parsed.item == {"key": "value", "other": "two"}


def test_dict_key_colliding_with_model_typed_field() -> None:
    """`<data><options>x</options>...` must not be built as `Filters`."""
    parsed = ModelCollisionXML.parse(
        "<tool><request>model_collision_xml</request>"
        "<data><options>x</options><other>y</other></data>"
        "<options><category>RAG</category></options></tool>"
    )
    assert parsed is not None
    assert parsed.data == {"options": "x", "other": "y"}
    assert parsed.options == Filters(category="RAG")


def test_agent_accepts_collision_shaped_tool_call() -> None:
    """The agent dispatch path must accept the collision cases too."""
    agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(), use_tools=True, use_functions_api=False)
    )
    agent.enable_message(CollidingTagsXML)
    text = (
        "<tool><request>colliding_tags_xml</request>"
        "<values><item>alpha</item></values>"
        "<item><key>value</key></item></tool>"
    )
    messages = agent.get_formatted_tool_messages(text)
    assert len(messages) == 1
    assert messages[0].values == ["alpha"]
    assert messages[0].item == {"key": "value"}


@pytest.mark.parametrize("include_request", [True, False])
def test_agent_parses_collection_arguments(include_request: bool) -> None:
    agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(), use_tools=True, use_functions_api=False)
    )
    agent.enable_message(SearchXML)
    original = SearchXML(
        filters={"category": "RAG"}, tags=[], options=Filters(category="RAG")
    )
    text = original.format_example()
    if not include_request:
        text = text.replace("  <request>search_xml</request>\n", "")
    assert agent.get_formatted_tool_messages(text) == [original]
    response = agent.agent_response(text)
    assert response is not None
    assert response.content == "RAG|0|RAG"


def test_root_with_one_field_remains_a_mapping() -> None:
    class DefaultsXML(XMLToolMessage):
        request: str = "defaults_xml"
        purpose: str = "Default-only tool"

    assert (
        DefaultsXML.parse("<tool><request>defaults_xml</request></tool>")
        == DefaultsXML()
    )
