# Structured Output

Available in Langroid since v0.24.0.

On supported LLMs, including recent OpenAI LLMs (GPT-4o and GPT-4o mini) and local LLMs served by compatible inference servers,
in particular, [vLLM](https://github.com/vllm-project/vllm) and [llama.cpp](https://github.com/ggerganov/llama.cpp), the decoding process can be constrained to ensure that the model's output adheres to a provided schema, 
improving the reliability of tool call generation and, in general, ensuring that the output can be reliably parsed and processed by downstream applications.

See [here](../tutorials/local-llm-setup.md/#setup-llamacpp-with-a-gguf-model-from-huggingface) for instructions for usage with `llama.cpp` and [here](../tutorials/local-llm-setup.md/#setup-vllm-with-a-model-from-huggingface) for `vLLM`.

Given a `ChatAgent` `agent` and a type `type`, we can define a strict copy of the agent as follows:
```python
strict_agent = agent[type]
```

We can use this to allow reliable extraction of typed values from an LLM with minimal prompting. For example, to generate typed values given `agent`'s current context, we can define the following:

```python
def typed_agent_response(
    prompt: str,
    output_type: type,
) -> Any:
    response = agent[output_type].llm_response_forget(prompt)
    return agent.from_ChatDocument(response, output_type)
```

We apply this in [test_structured_output.py](https://github.com/langroid/langroid/blob/main/tests/main/test_structured_output.py), in which we define types which describe
countries and their presidents:
```python
class Country(BaseModel):
    """Info about a country"""

    name: str = Field(..., description="Name of the country")
    capital: str = Field(..., description="Capital of the country")


class President(BaseModel):
    """Info about a president of a country"""

    country: Country = Field(..., description="Country of the president")
    name: str = Field(..., description="Name of the president")
    election_year: int = Field(..., description="Year of election of the president")


class PresidentList(BaseModel):
    """List of presidents of various countries"""

    presidents: List[President] = Field(..., description="List of presidents")
```
and show that `typed_agent_response("Show me an example of two Presidents", PresidentsList)` correctly returns a list of two presidents with *no* prompting describing the desired output format.

In addition to Pydantic models, `ToolMessage`s, and simple Python types are supported. For instance, `typed_agent_response("What is the value of pi?", float)` correctly returns $\pi$ to several decimal places.

The following two detailed examples show how structured output can be used to improve the reliability of the [chat-tree example](https://github.com/langroid/langroid/blob/main/examples/basic/chat-tree.py): [this](https://github.com/langroid/langroid/blob/main/examples/basic/chat-tree-structured.py) shows how we can use output formats to force the agent to make the correct tool call in each situation and [this](https://github.com/langroid/langroid/blob/main/examples/basic/chat-tree-structured-simple.py) shows how we can simplify by using structured outputs to extract typed intermediate values and expressing the control flow between LLM calls and agents explicitly.
