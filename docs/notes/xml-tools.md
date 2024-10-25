# XML-based Tools

Available in Langroid since v0.17.0.

[`XMLToolMessage`][langroid.agent.xml_tool_message.XMLToolMessage] is 
an abstract class for tools formatted using XML instead of JSON.
It has been mainly tested with non-nested tool structures.

For example in [test_xml_tool_message.py](https://github.com/langroid/langroid/blob/main/tests/main/test_xml_tool_message.py)
we define a CodeTool as follows (slightly simplified here):

```python
class CodeTool(XMLToolMessage):
    request: str = "code_tool"
    purpose: str = "Tool for writing <code> to a <filepath>"

    filepath: str = Field(
        ..., 
        description="The path to the file to write the code to"
    )

    code: str = Field(
        ..., 
        description="The code to write to the file", 
        verbatim=True
    )
```

Especially note how the `code` field has `verbatim=True` set in the `Field`
metadata. This will ensure that the LLM receives instructions to 

- enclose `code` field contents in a CDATA section, and 
- leave the `code` contents intact, without any escaping or other modifications.

Contrast this with a JSON-based tool, where newlines, quotes, etc
need to be escaped. LLMs (especially weaker ones) often "forget" to do the right 
escaping, which leads to incorrect JSON, and creates a burden on us to "repair" the
resulting json, a fraught process at best. Moreover, studies have shown that
requiring that an LLM return this type of carefully escaped code
within a JSON string can lead to a significant drop in the quality of the code
generated[^1].

[^1]: [LLMs are bad at returning code in JSON.](https://aider.chat/2024/08/14/code-in-json.html)


Note that tools/functions in OpenAI and related APIs are exclusively JSON-based, 
so in langroid when enabling an agent to use a tool derived from `XMLToolMessage`, 
we set these flags in `ChatAgentConfig`:

- `use_functions_api=False` (disables OpenAI functions/tools)
- `use_tools=True` (enables Langroid-native prompt-based tools)


See also the [`WriteFileTool`][langroid.agent.tools.file_tools.WriteFileTool] for a 
concrete example of a tool derived from `XMLToolMessage`. This tool enables an 
LLM to write content (code or text) to a file.

If you are using an existing Langroid `ToolMessage`, e.g. `SendTool`, you can
define your own subclass of `SendTool`, say `XMLSendTool`, 
inheriting from both `SendTool` and `XMLToolMessage`; see this
[example](https://github.com/langroid/langroid/blob/main/examples/basic/xml_tool.py)


