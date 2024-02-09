# Running the chainlit apps

In your Python virtual env, ensure you have 
installed `langroid` with the `chainlit` extra using, e.g.

```bash
pip install langroid[chainlit]
```

Or if you already have `langroid` installed, you can install the `chainlit` extra using:

```bash
pip install chainlit
```

To check that `chainlit` is installed, run:

```bash
chainlit hello
```

and you should see the `hello app` open in your browser.


## General usage
See [chainlit docs](https://docs.chainlit.io/get-started/overview) to learn the basics.

Generally speaking to use Langroid `ChatAgents` or `Tasks` with 
`chainlit`, you simply need to wrap your `ChatAgent` or `Task` in the appropriate 
"callback injection" class, e.g. either
```
import langroid as lr
agent = lr.ChatAgent(...)
lr.ChainlitAgentCallbacks(agent) 
```
or 
```
task = lr.Task(...)
lr.ChainlitTaskCallbacks(task) 
```
The `ChainlitTaskCallbacks` class recursively injects callbacks into 
`ChatAgents` belonging to the task, and any sub-tasks.
The callback classes are defined 
[here](https://github.com/langroid/langroid/blob/main/langroid/agent/callbacks/chainlit.py).

You also need to write an `on_chat_start` function and possibly an `on_message`
function to start off the app. See the examples to learn more.

## Configuration

You can configure some aspects of the chainlit app via these files,
which are included in this repo at the root level (see
the Chainlit [customization docs](https://docs.chainlit.io/customisation/overview) for more details):
- `.chainlit/config.toml` to customize project, features, UI (see [here](https://docs.chainlit.io/backend/config/overview))
- `.chainlit/translations/en-US.json` for various ["translations"](https://docs.chainlit.io/customisation/translation) and language-specific
   customizations. In particular, the default text in the input box is customized here.
- `chainlit.md`, which contains the initial "Readme" content
- [Logo, favicons](https://docs.chainlit.io/customisation/custom-logo-and-favicon) should be placed in a directory
  named `public` adjacent to the apps. 

Depending on how you organize your apps, you may need to run the `chainlit` command 
from the directory where the above customization files/dirs are placed.
