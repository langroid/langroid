from typing import Dict, List

import pytest
from pydantic import BaseModel

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
        return f"{self.filters.get('category', 'all')}|{len(self.tags)}|{self.options.category}"


@pytest.mark.parametrize("filters", [{}, {"category": "RAG"}, {"a": "A", "b": "B"}])
@pytest.mark.parametrize("tags", [[], ["AI"], ["AI", "RAG"]])
def test_xml_tool_collection_roundtrip(
    filters: Dict[str, str], tags: List[str]
) -> None:
    original = SearchXML(filters=filters, tags=tags, options=Filters(category="RAG"))
    parsed = SearchXML.parse(original.format_example())
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
