# Language Models: Completion vs Chat-Completion Endpoints

## Language Models as Next-token Predictors

A Language Model is essentially a "next-token prediction" model,
and so all LLMs today provide a "completion" endpoint.
The endpoint simply takes a prompt and returns a completion (i.e. a continuation).
A typical prompt sent to a completion endpoint might look like this:
```
The capital of Belgium is 
```
and the LLM will return a completion like this:
```
Brussels.
```
OpenAI's GPT3 is an example of a pure completion LLM.
But interacting with a completion LLM is not very natural or useful:
you would always need to formulate your input as a prompt
whose natural continuation is your desired output.
For example, if you wanted the LLM to highlight all proper nouns in a sentence,
you would format it as the following prompt:

**Example P:** Chat/Instruction converted to a completion prompt.

```
User: here is a sentence, the Assistant's task is to identify all proper nouns.
     Jack lives in Bosnia, and Jill lives in Belgium.
Assistant:    
```
The natural continuation of this prompt would be a response listing the proper nouns,
something like:
```
John, Bosnia, Jill, Belgium are all proper nouns.
```

This _seems_ sensible in theory, but a "bare" LLM that performs well on completions
may _not_ perform well on these kinds of prompts. The reason is that during its training, it may not
have been exposed to very many examples of this type of prompt-response pair.
So how can an LLM be improved to perform well on these kinds of prompts?

## Instruction-tuned, Aligned LLMs 

This brings us to the heart of the innovation behind the wildly popular ChatGPT:
it uses an enhancement of GPT3 that (besides having a lot more parameters),
was _explicitly_ fine-tuned on instructions (and dialogs more generally) -- this is referred to
as **instruction-fine-tuning** or IFT for short. In addition to fine-tuning on instructions/dialogs,
the models behind ChatGPT (i.e., GPT-3.5-Turbo and GPT-4) are further tuned to produce
responses that _align_ with human preferences (i.e. produce responses preferred by humans),
using a procedure called Reinforcement Learning with Human Feedback (RLHF).


For convenience, we refer to the combination of IFT and RLHF as **chat-tuning**.
A chat-tuned LLM can be expected to perform well on prompts such as the one in Example P above.
These types of prompts are still unnatural, however, so as a convenience,
chat-tuned LLM API servers also provide a "chat-completion" endpoint, which allows the user
to interact with them in a natural dialog, which might look like this
(the portions in square brackets are indicators of who is generating the text):

```
[User] What is the capital of Belgium?
[Assistant] The capital of Belgium is Brussels.
```
or
```
[User] In the text below, find all proper nouns:
    Jack lives in Bosnia, and Jill lives in Belgium.
[Assistant] John, Bosnia, Jill, Belgium are all proper nouns.
[User] where does John live?
[Assistant] John lives in Bosnia.
```

## Chat Completion Endpoints: under the hood

How could this work, given that LLMs are fundamentally next-token predictors?
This is a convenience provided by the LLM API service (e.g. OpenAI or the APIs
of local models):
when a user invokes the chat-completion endpoint (typically
at `/chat/completion` under the base URL), under the hood, the server converts the
instructions and multi-turn chat history into a single string, with annotations indicating
user and assistant turns, and ending with something like "Assistant:"
as in the Example P above.

Now the subtle detail to note here is that it matters _how_ the
dialog (instructions plus chat history) is converted into a single prompt string.
Converting to a single prompt by simply concatenating the
instructions and chat history using an "intuitive" format (e.g. indicating
user, assistant turns using "User", "Assistant:", etc.) _can_ work,
however most local LLMs are trained on a _specific_ prompt format.
So if we format chats in a different way, we may get odd/inferior results.

## Converting Chats to Prompts: Formatting Rules

For example, the llama2 models are trained on a format where the user's input is bracketed within special strings `[INST]`
and `[/INST]`. There are other requirements that we don't go into here, but
interested readers can refer to these links:

- A reddit thread on the [llama2 formats](https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/)
- Facebook's [llama2 code](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L44)
- Langroid's [llama2 formatting code](https://github.com/langroid/langroid/blob/main/langroid/language_models/prompt_formatter/llama2_formatter.py)

A dialog fed to a Llama2 model in its expected prompt format would look like this:

```
<s>[INST] <<SYS>>
You are are a helpful assistant.
<</SYS>>

Hi there! 
[/INST] 
Hello! How can I help you today? </s>
<s>[INST] In the text below, find all proper nouns:
    Jack lives in Bosnia, and Jill lives in Belgium.
 [/INST] 
John, Bosnia, Jill, Belgium are all proper nouns. </s><s> 
[INST] Where does Jack live? [/INST] 
Jack lives in Bosnia. </s><s>
[INST] And Jill? [/INST]
Jill lives in Belgium. </s><s>
[INST] Which are its neighboring countries? [/INST]
```

This means that if a library wants to provide a chat-completion endpoint for
a local model, it needs to provide a way to convert chat history to a single prompt
using the specific formatting rules of the model.
The `ooba` (`text-generation-webui`) library has an extensive set of chat formatting
templates for a variety of models, and their model server auto-detects the
format template from the model name.

A user of these local LLM server libraries thus has two options when using a local in chat mode:

- use the _chat-completion_ endpoint, and let the underlying library handle the chat-to-prompt formatting, or
- first format the chat history according to the model's requirements, and then use the
  _completion_ endpoint

## Using Local Models in Langroid

Local models can be used in Langroid by defining a `LocalModelConfig` object.
More details are in this [tutorial](tutorials/local-llm.md), but here we briefly
discuss prompt-formatting in this context.
Langroid provides a built-in [formatter for LLama2 models](https://github.com/langroid/langroid/blob/main/langroid/language_models/prompt_formatter/llama2_formatter.py), 
so users looking to use llama2 models with langroid can try either of these options, by setting the
`use_completion_for_chat` flag in the `LocalModelConfig` object
(See the local-LLM [tutorial](tutorials/local-llm.md) for details).
When this flag is set to `True`, the chat history is formatted using the built-in llama2 formatter
and the completion endpoint is used. 
When the flag is set to `False`, the chat history is sent directly to the chat-completion
endpoint, which internally converts the chat history to a prompt in the expected llama2 format.

For local models other than Llama2, users can write their own formatters by
writing a class similar to `Llama2Formatter` and then setting the `use_completion_for_chat` flag
to `False` in the `LocalModelConfig` object.




