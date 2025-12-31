import json
import os

import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)


@pytest.mark.openai_responses
@pytest.mark.slow
class TestStructuredOutput:
    def test_json_schema_response(self):
        """Model returns valid JSON matching schema."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from pydantic import BaseModel, Field

        from langroid.language_models.base import LLMFunctionSpec, OpenAIJsonSchemaSpec

        class PersonInfo(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")
            city: str = Field(description="Person's city")

        # Create schema from Pydantic model
        schema_spec = OpenAIJsonSchemaSpec(
            strict=True,
            function=LLMFunctionSpec(
                name="person_info",
                description="Person information",
                parameters=PersonInfo.model_json_schema(),
            ),
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content="Generate info for John, 30 years old, from Paris as JSON.",
            ),
        ]

        response = llm.chat(messages, response_format=schema_spec, max_tokens=100)

        # Response should be valid JSON
        data = json.loads(response.message)

        assert "name" in data
        assert "John" in data["name"]
        assert data["age"] == 30
        assert "Paris" in data["city"]

    def test_json_object_response_format(self):
        """Test simple json_object response format."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        # Simple JSON object format (not strict schema)
        # For simple json_object, we can just use a dict
        json_format = {"type": "json_object"}

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content="List three colors with their hex codes as JSON",
            ),
        ]

        response = llm.chat(messages, response_format=json_format, max_tokens=100)

        # Should be valid JSON
        data = json.loads(response.message)
        assert isinstance(data, dict)
        # Should have some color data
        assert len(data) > 0

    def test_complex_nested_schema(self):
        """Test complex nested JSON schema."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from typing import List

        from pydantic import BaseModel, Field

        from langroid.language_models.base import LLMFunctionSpec, OpenAIJsonSchemaSpec

        class Address(BaseModel):
            street: str = Field(description="Street address")
            city: str = Field(description="City")
            country: str = Field(description="Country")

        class Company(BaseModel):
            name: str = Field(description="Company name")
            employees: int = Field(description="Number of employees")
            address: Address = Field(description="Company address")
            departments: List[str] = Field(description="List of departments")

        schema_spec = OpenAIJsonSchemaSpec(
            strict=True,
            function=LLMFunctionSpec(
                name="company_info",
                description="Company information",
                parameters=Company.model_json_schema(),
            ),
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content=(
                    "Create JSON for tech company TechCorp with 500 employees, "
                    "located at 123 Silicon Way, San Francisco, USA. "
                    "Include Engineering, Sales, and Marketing departments."
                ),
            ),
        ]

        response = llm.chat(messages, response_format=schema_spec, max_tokens=200)

        # Parse and validate
        data = json.loads(response.message)

        assert data["name"] == "TechCorp"
        assert data["employees"] == 500
        assert data["address"]["street"] == "123 Silicon Way"
        assert data["address"]["city"] == "San Francisco"
        assert data["address"]["country"] == "USA"
        assert "Engineering" in data["departments"]
        assert "Sales" in data["departments"]
        assert "Marketing" in data["departments"]

    def test_streaming_with_json_schema(self):
        """Test that streaming works with JSON schema."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from pydantic import BaseModel, Field

        from langroid.language_models.base import LLMFunctionSpec, OpenAIJsonSchemaSpec

        class SimpleResponse(BaseModel):
            status: str = Field(description="Status message")
            value: int = Field(description="Numeric value")

        schema_spec = OpenAIJsonSchemaSpec(
            strict=True,
            function=LLMFunctionSpec(
                name="simple_response",
                description="Simple response format",
                parameters=SimpleResponse.model_json_schema(),
            ),
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=True,  # Enable streaming
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content="Return JSON with status='success' and value=42",
            ),
        ]

        response = llm.chat(messages, response_format=schema_spec, max_tokens=50)

        # Even with streaming, should get valid JSON
        data = json.loads(response.message)
        assert data["status"] == "success"
        assert data["value"] == 42

    def test_response_format_with_tools(self):
        """Test that response_format works alongside tools."""
        if os.getenv("OPENAI_API_KEY", "") == "":
            pytest.skip("OPENAI_API_KEY not set; skipping real API test")

        from pydantic import BaseModel, Field

        from langroid.language_models.base import (
            LLMFunctionSpec,
            OpenAIJsonSchemaSpec,
            OpenAIToolSpec,
        )

        class QueryResult(BaseModel):
            query: str = Field(description="The search query")
            found: bool = Field(description="Whether results were found")
            count: int = Field(description="Number of results")

        schema_spec = OpenAIJsonSchemaSpec(
            strict=True,
            function=LLMFunctionSpec(
                name="query_result",
                description="Query result format",
                parameters=QueryResult.model_json_schema(),
            ),
        )

        search_tool = OpenAIToolSpec(
            type="function",
            function={
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        )

        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4.1"),
            stream=False,
            temperature=0.2,
        )
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content=(
                    "Return JSON with query='weather in Paris', found=true, count=10. "
                    "Do NOT call the search tool, just return the JSON."
                ),
            ),
        ]

        # Should return JSON, not call tool (due to explicit instruction)
        response = llm.chat(
            messages, tools=[search_tool], response_format=schema_spec, max_tokens=100
        )

        # Should have JSON response, not tool call
        assert response.message
        data = json.loads(response.message)
        assert data["query"] == "weather in Paris"
        assert data["found"] is True
        assert data["count"] == 10
