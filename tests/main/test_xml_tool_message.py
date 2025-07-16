from typing import Dict, List, Tuple

import pytest

import langroid as lr
from langroid.agent.tools.orchestration import ResultTool
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.exceptions import XMLException
from langroid.pydantic_v1 import BaseModel, Field
from langroid.utils.configuration import Settings, set_global


class CodeTool(XMLToolMessage):
    request: str = "code_tool"
    purpose: str = "Tool for writing <code> with a certain <version> to a <filepath>"

    filepath: str = Field(..., description="The path to the file to write the code to")
    version: int = Field(..., description="The version number of the code")
    # NOTE: we are setting a custom attrib verbatim to True to indicate that
    # parsing/formatting should be verbatim, and to ensure that LLM is instructed
    # to enclose the content in a CDATA section
    code: str = Field(..., description="The code to write to the file", verbatim=True)

    @classmethod
    def examples(cls) -> List[XMLToolMessage | Tuple[str, XMLToolMessage]]:
        return [
            (
                "I want to create a new Python file with a simple print statement",
                cls(
                    filepath="/path/to/new_file.py",
                    version=1,
                    code='print("Hello from CodeTool!")',
                ),
            ),
            cls(
                filepath="/path/to/existing_file.py",
                version=2,
                code='def greet(name):\n    print(f"Hello, {name}!")\n\ngreet("World")',
            ),
        ]

    def handle(self) -> ResultTool:
        return ResultTool(
            filepath=self.filepath,
            version=self.version,
            code=self.code,
        )


def test_find_candidates():
    root_tag = CodeTool.Config.root_element
    text = f"""
    Some text before
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
        <code><![CDATA[
print("Hello, World!")
]]></code>
    </{root_tag}>
    Some text in between
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/another.py</filepath>
        <version>2</version>
        <code><![CDATA[def greet(): 
    print("Hi!")]]></code>
    </{root_tag}>
    Some text after
    """
    candidates = CodeTool.find_candidates(text)
    assert len(candidates) == 2
    for candidate in candidates:
        assert isinstance(CodeTool.parse(candidate), CodeTool)


def test_find_candidates_missing_closing_tag():
    root_tag = CodeTool.Config.root_element
    text = f"""
    Some text before
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
        <code><![CDATA[print("Hello, World!")]]></code>
    </{root_tag}>
    Some text in between
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/another.py</filepath>
        <version>2</version>
        <code><![CDATA[def greet(): 
    print("Hi!")]]></code>
    Some text after
    """
    candidates = CodeTool.find_candidates(text)
    assert len(candidates) == 2
    for candidate in candidates:
        assert isinstance(CodeTool.parse(candidate), CodeTool)


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("<tool><field1>data</field1></tool>", ["<tool><field1>data</field1></tool>"]),
        (  # missing open tag
            "Hello <field1>data</field1></tool>",
            ["<tool><field1>data</field1></tool>"],
        ),
        (  # proper open/close tags
            "<tool>a</tool> stuff <tool>b</tool>",
            ["<tool>a</tool>", "<tool>b</tool>"],
        ),
        ("just plain text", []),
        (
            # allow missing closing tag for last element
            "<tool><field1>data</field1>",
            ["<tool><field1>data</field1></tool>"],
        ),
    ],
)
def test_find_candidates_tolerant(input_text, expected):
    # check that missing opening tag is tolerated, and other cases
    class TestXMLTool(XMLToolMessage):
        field1: str
        field2: str

    assert TestXMLTool.find_candidates(input_text) == expected


def test_parse():
    root_tag = CodeTool.Config.root_element
    xml_string = f"""
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
        <code><![CDATA[
```
print("Hello, World!")
```
]]></code>
    </{root_tag}>
    """
    code_tool = CodeTool.parse(xml_string)
    assert isinstance(code_tool, CodeTool)
    assert code_tool.request == "code_tool"
    assert code_tool.filepath == "/path/to/file.py"
    assert code_tool.version == 1
    assert code_tool.code == 'print("Hello, World!")'


def test_parse_bad_format():
    root_tag = CodeTool.Config.root_element
    # test with missing closing tag
    bad_xml_string = f"""
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
        <code>
            print("Hello, World!")
    </{root_tag}>
    """
    with pytest.raises(XMLException):
        CodeTool.parse(bad_xml_string)

    # Test with missing required field
    incomplete_xml_string = f"""
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
    </{root_tag}>
    """
    with pytest.raises(XMLException):
        CodeTool.parse(incomplete_xml_string)

    # Test with invalid XML structure
    invalid_xml_string = f"""
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
        <code><![CDATA[print("Hello, World!")]]></code>
    </{root_tag}
    """
    with pytest.raises(XMLException):
        CodeTool.parse(invalid_xml_string)


def test_format():
    root_tag = CodeTool.Config.root_element
    code_tool = CodeTool(
        filepath="/path/to/file.py",
        version=1,
        code='print("Hello, World!")',
    )
    formatted = code_tool.format_example()
    assert f"<{root_tag}>" in formatted
    assert "<request>code_tool</request>" in formatted
    assert "<filepath>/path/to/file.py</filepath>" in formatted
    assert "<version>1</version>" in formatted
    assert '<code><![CDATA[print("Hello, World!")]]></code>' in formatted
    assert f"</{root_tag}>" in formatted


def test_roundtrip():
    original = CodeTool(
        filepath="/path/to/file.py",
        version=1,
        code='print("Hello, World!")',
    )
    formatted = original.format_example()
    parsed = CodeTool.parse(formatted)
    assert original.model_dump() == parsed.model_dump()


def test_tolerant_parsing():
    root_tag = CodeTool.Config.root_element
    messy_xml_string = f"""
    <{root_tag}>
        <request>
            code_tool
        </request>
        <filepath>
            /path/to/file.py
        </filepath>
        <version>
            1
        </version>
        <code><![CDATA[
def hello():
    print("Hello, World!")

hello()
        ]]></code>
    </{root_tag}>
    """
    code_tool = CodeTool.parse(messy_xml_string)

    assert isinstance(code_tool, CodeTool)
    assert code_tool.request.strip() == "code_tool"
    assert code_tool.filepath.strip() == "/path/to/file.py"
    assert code_tool.version == 1

    expected_code = """
def hello():
    print("Hello, World!")

hello()
""".strip()
    assert code_tool.code.strip() == expected_code


def test_instructions():
    instructions = CodeTool.format_instructions()
    root_tag = CodeTool.Config.root_element

    assert "Placeholders:" in instructions
    assert "FILEPATH = [value for filepath]" in instructions
    assert "VERSION = [value for version]" in instructions
    assert "CODE = [value for code]" in instructions
    assert "REQUEST = [value for request]" in instructions

    assert "Formatting example:" in instructions
    assert f"<{root_tag}>" in instructions
    assert f"</{root_tag}>" in instructions
    assert "<filepath>{FILEPATH}</filepath>" in instructions
    assert "<version>{VERSION}</version>" in instructions
    assert "<code><![CDATA[{CODE}]]></code>" in instructions
    assert "<request>{REQUEST}</request>" in instructions


def test_llm_xml_tool_message(
    test_settings: Settings,
):
    set_global(test_settings)
    code_tool_name = CodeTool.default_value("request")

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="TestAgent",
            use_functions_api=False,
            use_tools=True,
            system_message=f"""
            When asked to write Python code, 
            you must use the TOOL `{code_tool_name}` to complete this task.
            """,
        )
    )
    agent.enable_message(CodeTool)
    task = lr.Task(agent, interactive=False)[ResultTool]
    result = task.run(
        """
        Write a simple python function that takes a name as string arg, 
        and prints hello to that name.
        Write the code to the file src/mycode.py, with version number 7
        """
    )
    assert isinstance(result, ResultTool)
    assert result.filepath == "src/mycode.py"
    assert result.version == 7
    assert all(word in result.code.lower() for word in ["def", "hello", "print"])

    result = task.run(
        """
        Write a Rust function to calculate the n'th fibonacci number,
        and add a test block. Write it to the file src/fib.rs, with version number 3
        """
    )
    assert isinstance(result, ResultTool)
    assert result.filepath == "src/fib.rs"
    assert result.version == 3
    assert all(word in result.code.lower() for word in ["fn", "fibonacci", "test"])


class Address(BaseModel):
    # declare street as verbatim, to test that the formatting encloses
    # the value in a CDATA block
    street: str = Field(..., description="The street address", verbatim=True)
    city: str
    country: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


class ComplexNestedXMLTool(XMLToolMessage):
    request: str = "complex_nested_tool"
    purpose: str = "To present a complex nested structure"

    person: Person
    hobbies: List[str]
    phones: Dict[str, int]
    friends: List[Person] | None = None

    @classmethod
    def examples(cls) -> List[XMLToolMessage | Tuple[str, XMLToolMessage]]:
        return [
            (
                "I want to present a person named John Doe, aged 30, "
                "living at 123 Main St, Anytown, USA, with hobbies of "
                "reading and cycling, "
                " phone numbers: home (1234567890) and work (9876543210)"
                " and two friends: "
                "   Jane Doe, aged 28, living at 456 Elm St, Somewhere, Canada, "
                "   Jack Doe, aged 32, living at 789 Oak St, Anywhere, UK",
                cls(
                    person=Person(
                        name="John Doe",
                        age=30,
                        address=Address(
                            street="123 Main St", city="Anytown", country="USA"
                        ),
                    ),
                    hobbies=["reading", "cycling"],
                    phones={"home": 1234567890, "work": 9876543210},
                    friends=[
                        Person(
                            name="Jane Doe",
                            age=28,
                            address=Address(
                                street="456 Elm St", city="Somewhere", country="Canada"
                            ),
                        ),
                        Person(
                            name="Jack Doe",
                            age=32,
                            address=Address(
                                street="789 Oak St", city="Anywhere", country="UK"
                            ),
                        ),
                    ],
                ),
            )
        ]

    def handle(self) -> ResultTool:
        return ResultTool(
            person=self.person,
            hobbies=self.hobbies,
            phones=self.phones,
            friends=self.friends,
        )


@pytest.fixture
def complex_nested_xml_tool():
    return ComplexNestedXMLTool(
        person=Person(
            name="Jane Doe",
            age=28,
            address=Address(street="456 Elm St", city="Somewhere", country="Canada"),
        ),
        hobbies=["painting", "hiking"],
        phones={"mobile": 5551234567, "work": 5559876543},
        friends=[
            Person(
                name="John Doe",
                age=30,
                address=Address(street="123 Main St", city="Anytown", country="USA"),
            ),
            Person(
                name="Jack Doe",
                age=32,
                address=Address(street="789 Oak St", city="Anywhere", country="UK"),
            ),
        ],
    )


def test_format_complex_nested(complex_nested_xml_tool: ComplexNestedXMLTool):
    complex_tool = complex_nested_xml_tool
    formatted = complex_tool.format_example()
    print(formatted)  # For debugging
    assert "<person>" in formatted
    assert "<name>Jane Doe</name>" in formatted
    assert "<age>28</age>" in formatted
    assert "<address>" in formatted
    # NOTE: street was declared as verbatim, so it should be in a CDATA section
    assert "<street><![CDATA[456 Elm St]]></street>" in formatted
    assert "<city>Somewhere</city>" in formatted
    assert "<country>Canada</country>" in formatted
    assert "<hobbies>" in formatted
    assert "<item>painting</item>" in formatted
    assert "<item>hiking</item>" in formatted
    assert "<phones>" in formatted
    assert "<mobile>5551234567</mobile>" in formatted
    assert "<work>5559876543</work>" in formatted
    assert "<friends>" in formatted
    assert "<person>" in formatted
    assert "<name>John Doe</name>" in formatted
    assert "<age>30</age>" in formatted
    assert "<name>Jack Doe</name>" in formatted
    assert "<age>32</age>" in formatted

    # Test case for absent friends field
    complex_tool_no_friends = ComplexNestedXMLTool(
        person=Person(
            name="Alice Smith",
            age=25,
            address=Address(street="789 Pine St", city="Nowhere", country="USA"),
        ),
        hobbies=["reading", "swimming"],
        phones={"home": 1234567890},
        friends=None,
    )
    formatted_no_friends = complex_tool_no_friends.format_example()
    print(formatted_no_friends)  # For debugging
    assert "<friends>" not in formatted_no_friends


def test_parse_complex_nested():
    xml_string = """
    <tool>
        <request>complex_nested_tool</request>
        <person>
            <name>John Doe</name>
            <age>30</age>
            <address>
                <street>123 Main St</street>
                <city>Anytown</city>
                <country>USA</country>
            </address>
        </person>
        <hobbies>
            <item>reading</item>
            <item>cycling</item>
        </hobbies>
        <phones>
            <home>1234567890</home>
            <work>9876543210</work>
        </phones>
    </tool>
    """
    parsed = ComplexNestedXMLTool.parse(xml_string)
    assert isinstance(parsed, ComplexNestedXMLTool)
    assert parsed.request == "complex_nested_tool"
    assert isinstance(parsed.person, Person)
    assert parsed.person.name == "John Doe"
    assert parsed.person.age == 30
    assert isinstance(parsed.person.address, Address)
    assert parsed.person.address.street == "123 Main St"
    assert parsed.person.address.city == "Anytown"
    assert parsed.person.address.country == "USA"
    assert parsed.hobbies == ["reading", "cycling"]
    assert parsed.phones == {"home": 1234567890, "work": 9876543210}


def test_instructions_complex_nested():
    instructions = ComplexNestedXMLTool.format_instructions()
    root_tag = ComplexNestedXMLTool.Config.root_element

    assert "Placeholders:" in instructions
    assert "REQUEST = [value for request]" in instructions
    assert "PERSON = [nested structure for person]" in instructions
    assert "NAME = [value for name]" in instructions
    assert "AGE = [value for age]" in instructions
    assert "ADDRESS = [nested structure for address]" in instructions
    assert "STREET = [value for street]" in instructions
    assert "CITY = [value for city]" in instructions
    assert "COUNTRY = [value for country]" in instructions
    assert "HOBBIES = [list of str for hobbies]" in instructions
    assert "PHONES = [dictionary with str keys and int values]" in instructions
    assert "FRIENDS = [list of nested structures for friends]" in instructions

    assert "Formatting example:" in instructions
    assert f"<{root_tag}>" in instructions
    assert f"</{root_tag}>" in instructions
    assert "<request>{REQUEST}</request>" in instructions
    assert "<person>" in instructions
    assert "<name>{NAME}</name>" in instructions
    assert "<age>{AGE}</age>" in instructions
    assert "<address>" in instructions
    # NOTE: street was declared as verbatim, so it should be in a CDATA section
    assert "<street><![CDATA[{STREET}]]></street>" in instructions
    assert "<city>{CITY}</city>" in instructions
    assert "<country>{COUNTRY}</country>" in instructions
    assert "</address>" in instructions
    assert "</person>" in instructions
    assert "<hobbies>" in instructions
    assert "<item>[str value]</item>" in instructions
    assert "</hobbies>" in instructions
    assert "<phones>" in instructions
    assert "<str>[int value]</str>" in instructions
    assert "</phones>" in instructions
    assert "<friends>" in instructions
    assert "<item>[Person value]</item>" in instructions
    assert "</friends>" in instructions


def test_roundtrip_complex_nested(complex_nested_xml_tool):
    original = complex_nested_xml_tool

    formatted = original.format_example()
    parsed = ComplexNestedXMLTool.parse(formatted)
    assert original.model_dump() == parsed.model_dump()

    # Additional checks for nested structures
    assert original.person.model_dump() == parsed.person.model_dump()
    assert original.person.address.model_dump() == parsed.person.address.model_dump()
    assert original.hobbies == parsed.hobbies
    assert original.phones == parsed.phones


def test_roundtrip_complex_nested_tolerant():
    # note there is no `friends` field, so this is a good test
    # to check that the formatting is not including this field in the XML.
    original = ComplexNestedXMLTool(
        person=Person(
            name="Jane Doe",
            age=28,
            address=Address(street="456 Elm St", city="Somewhere", country="Canada"),
        ),
        hobbies=["painting", "hiking"],
        phones={"mobile": 5551234567, "work": 5559876543},
    )
    formatted = original.format_example()

    # Insert harmless whitespace
    formatted_with_whitespace = (
        formatted.replace("<", " \n <").replace(">", "> \n ").replace("</", " \n </")
    )

    parsed = ComplexNestedXMLTool.parse(formatted_with_whitespace)

    assert original.model_dump() == parsed.model_dump()
    assert original.person.model_dump() == parsed.person.model_dump()
    assert original.person.address.model_dump() == parsed.person.address.model_dump()
    assert original.hobbies == parsed.hobbies
    assert original.phones == parsed.phones


def test_llm_complex_xml_tool_message(
    test_settings: Settings,
):
    set_global(test_settings)
    complex_tool_name = ComplexNestedXMLTool.default_value("request")

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="TestAgent",
            use_functions_api=False,
            use_tools=True,
            system_message=f"""
            When asked to provide information about a person,
            you must use the TOOL `{complex_tool_name}` to complete this task.
            """,
        )
    )
    agent.enable_message(ComplexNestedXMLTool)
    task = lr.Task(agent, interactive=False)[ResultTool]
    result = task.run(
        """
        Provide information about a person named Alice Johnson, aged 35,
        living at 789 Oak Ave, Springfield, USA, with hobbies of
        gardening and cooking, and phone numbers: 
        home (5551112222) and mobile (5553334444).
        Also include information about her two friends:
        1. Bob Smith, aged 40, living at 123 Maple St, Riverside, USA
        2. Carol White, aged 38, living at 456 Pine Rd, Hillside, USA
        """
    )
    assert isinstance(result, ResultTool)
    assert isinstance(result.person, Person)
    assert result.person.name == "Alice Johnson"
    assert result.person.age == 35
    assert isinstance(result.person.address, Address)
    assert result.person.address.street == "789 Oak Ave"
    assert result.person.address.city == "Springfield"
    assert result.person.address.country == "USA"
    assert set(result.hobbies) == {"gardening", "cooking"}
    assert result.phones == {"home": 5551112222, "mobile": 5553334444}
    assert isinstance(result.friends, list)
    assert len(result.friends) == 2
    assert result.friends[0].name == "Bob Smith"
    assert result.friends[0].age == 40
    assert result.friends[0].address.street == "123 Maple St"
    assert result.friends[0].address.city == "Riverside"
    assert result.friends[0].address.country == "USA"
    assert result.friends[1].name == "Carol White"
    assert result.friends[1].age == 38
    assert result.friends[1].address.street == "456 Pine Rd"
    assert result.friends[1].address.city == "Hillside"
    assert result.friends[1].address.country == "USA"


if __name__ == "__main__":
    pytest.main([__file__])
