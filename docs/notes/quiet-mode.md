# Suppressing LLM output: quiet mode

In some scenarios we want to suppress LLM streaming output -- e.g. when doing some type of processing as part of a workflow,
or when using an LLM-agent to generate code via tools, etc. We are more interested in seeing the results of the workflow,
and don't want to see streaming output in the terminal. Langroid provides a `quiet_mode` context manager that can be used
to suppress LLM output, even in streaming mode (in fact streaming is disabled in quiet mode).

E.g.  we can use the `quiet_mode` context manager like this:

```python
from langroid.utils.configuration import quiet_mode, settings

# directly with LLM

llm = ...
with quiet_mode(True):
	response = llm.chat(...)

# or, using an agent

agent = ...
with quiet_mode(True):
	response = agent.llm_response(...)

# or, using a task

task = Task(agent, ...)
with quiet_mode(True):
	result = Taks.run(...)

# we can explicitly set quiet_mode, and this is globally recognized throughout langroid.

settings.quiet = True

# we can also condition quiet mode on another custom cmd line option/flag, such as "silent":

with quiet_mode(silent):
	...

```