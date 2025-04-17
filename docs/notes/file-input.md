# PDF Files and Image inputs to LLMs

Langroid supports sending PDF files and images (either URLs or local files)
directly to Large Language Models with multi-modal 
capabilities. This feature allows models to "see" files and other documents,
and works with most multi-modal models served via an OpenAI-compatible API,
e.g.:

- OpenAI's GPT-4o series and GPT-4.1 series
- Gemini models
- Claude series models (via OpenAI-compatible providers like OpenRouter or LiteLLM )

To see example usage, see:

- tests: [test_llm.py](https://github.com/langroid/langroid/blob/main/tests/main/test_llm.py), 
   [test_llm_async.py](https://github.com/langroid/langroid/blob/main/tests/main/test_llm_async.py),
   [test_chat-agent.py](https://github.com/langroid/langroid/blob/main/tests/main/test_chat_agent.py).
- example script: [pdf-json-no-parse.py](https://github.com/langroid/langroid/blob/main/examples/extract/pdf-json-no-parse.py), which shows
  how you can directly extract structured information from a document 
  **without having to first parse it to markdown** (which is inherently lossy).

## Basic Usage directly with LLM `chat` and `achat` methods

First create a `FileAttachment` object using one of the `from_` methods.
For image (`png`, `jpg/jpeg`) files you can use `FileAttachment.from_path(p)`
where `p` is either a local file path, or a http/https URL.
For PDF files, you can use `from_path` with a local file, or `from_bytes` or `from_io`
(see below). In the examples below we show only `pdf` examples.

```python
from langroid.language_models.base import LLMMessage, Role
from langroid.parsing.file_attachment import FileAttachment
import langroid.language_models as lm

# Create a file attachment
attachment = FileAttachment.from_path("path/to/document.pdf")

# Create messages with attachment
messages = [
    LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
    LLMMessage(
        role=Role.USER, content="What's the title of this document?", 
        files=[attachment]
    )
]

# Set up LLM with model that supports attachments
llm = lm.OpenAIGPT(lm.OpenAIGPTConfig(chat_model=lm.OpenAIChatModel.GPT4o))

# Get response
response = llm.chat(messages=messages)
```

## Supported File Formats

Currently the OpenAI-API supports:

- PDF files (including image-based PDFs)
- image files and URLs


## Creating Attachments

There are multiple ways to create file attachments:

```python
# From a file path
attachment = FileAttachment.from_path("path/to/file.pdf")

# From bytes
with open("path/to/file.pdf", "rb") as f:
    attachment = FileAttachment.from_bytes(f.read(), filename="document.pdf")

# From a file-like object
from io import BytesIO
file_obj = BytesIO(pdf_bytes)
attachment = FileAttachment.from_io(file_obj, filename="document.pdf")
```

## Follow-up Questions

You can continue the conversation with follow-up questions that reference the attached files:

```python
messages.append(LLMMessage(role=Role.ASSISTANT, content=response.message))
messages.append(LLMMessage(role=Role.USER, content="What is the main topic?"))
response = llm.chat(messages=messages)
```

## Multiple Attachments

Langroid allows multiple files can be sent in a single message,
but as of 16 Apr 2025, sending multiple PDF files does not appear to be properly supported in the 
APIs (they seem to only use the last file attached), although sending multiple 
images does work. 

```python
messages = [
    LLMMessage(
        role=Role.USER,
        content="Compare these documents",
        files=[attachment1, attachment2]
    )
]
```

## Using File Attachments with Agents

Agents can process file attachments as well, in the `llm_response` method,
which takes a `ChatDocument` object as input. 
To pass in file attachments, include the `files` field in the `ChatDocument`,
in addition to the content:

```python
import langroid as lr
from langroid.agent.chat_document import ChatDocument, ChatDocMetaData
from langroid.mytypes import Entity


agent = lr.ChatAgent(lr.ChatAgentConfig())

user_input = ChatDocument(
    content="What is the title of this document?",
    files=[attachment],
    metadata=ChatDocMetaData(
        sender=Entity.USER,
    )
)
# or more simply, use the agent's `create_user_response` method:
# user_input = agent.create_user_response(
#     content="What is the title of this document?",
#     files=[attachment],    
# )
response = agent.llm_response(user_input)
```


## Using File Attachments with Tasks

In Langroid,  `Task.run()` can take a `ChatDocument` object as input,
and as mentioned above, it can contain attached files in the `files` field.
To ensure proper orchestration, you'd want to properly set various `metadata` fields
as well, such as `sender`, etc. Langroid provides a convenient 
`create_user_response` method to create a `ChatDocument` object with the necessary 
metadata, so you only need to specify the `content` and `files` fields:


```python
from langroid.parsing.file_attachment import FileAttachment
from langroid.agent.task import Task

agent = ...
# Create task
task = Task(agent, interactive=True)

# Create a file attachment
attachment = FileAttachment.from_path("path/to/document.pdf")

# Create input with attachment
input_message = agent.create_user_response(
    content="Extract data from this document",
    files=[attachment]
)

# Run task with file attachment
result = task.run(input_message)
```

See the script [`pdf-json-no-parse.py`](https://github.com/langroid/langroid/blob/main/examples/extract/pdf-json-no-parse.py)
for a complete example of using file attachments with tasks.

## Practical Applications

- PDF document analysis and data extraction
- Report summarization
- Structured information extraction from documents
- Visual content analysis

For more complex applications, consider using the Task and Agent infrastructure in 
Langroid to orchestrate multi-step document processing workflows.