from typing import List, Tuple

import pytest

import langroid as lr
from langroid.agent.tools.orchestration import ResultTool
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.utils.configuration import Settings, set_global


class CodeTool(XMLToolMessage):
    request: str = "code_tool"
    purpose: str = "Tool for writing <code> with a certain <version> to a <filepath>"

    filepath: str
    version: int
    code: str

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
        <code><![CDATA[print("Hello, World!")]]></code>
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


def test_parse():
    root_tag = CodeTool.Config.root_element
    xml_string = f"""
    <{root_tag}>
        <request>code_tool</request>
        <filepath>/path/to/file.py</filepath>
        <version>1</version>
        <code><![CDATA[print("Hello, World!")]]></code>
    </{root_tag}>
    """
    code_tool = CodeTool.parse(xml_string)
    assert isinstance(code_tool, CodeTool)
    assert code_tool.request == "code_tool"
    assert code_tool.filepath == "/path/to/file.py"
    assert code_tool.version == 1
    assert code_tool.code == 'print("Hello, World!")'


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
    assert original.dict() == parsed.dict()


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


if __name__ == "__main__":
    pytest.main([__file__])
