# Multi-Agent collaboration via Task Delegation

## Why multiple agents?

Let's say we want to develop a complex LLM-based application, for example an application
that reads a legal contract, extracts structured information, cross-checks it against
some taxonomoy, gets some human input, and produces clear summaries.
In _theory_ it may be possible to solve this in a monolithic architecture using an
LLM API and a vector-store. But this approach
quickly runs into problems -- you would need to maintain multiple LLM conversation
histories and states, multiple vector-store instances, and coordinate all of the
interactions between them.

Langroid's `ChatAgent` and `Task` abstractions provide a natural and intuitive
way to decompose a solution approach
into multiple tasks, each requiring different skills and capabilities.
Some of these tasks may need access to an LLM,
others may need access to a vector-store, and yet others may need
tools/plugins/function-calling capabilities, or any combination of these.
It may also make sense to have some tasks that manage the overall solution process.
From an architectural perspective, this type of modularity has numerous benefits:

- **Reusability**: We can reuse the same agent/task in other contexts,
- **Scalability**: We can scale up the solution by adding more agents/tasks,
- **Flexibility**: We can easily change the solution by adding/removing agents/tasks.
- **Maintainability**: We can maintain the solution by updating individual agents/tasks.
- **Testability**: We can test/debug individual agents/tasks in isolation.
- **Composability**: We can compose agents/tasks to create new agents/tasks.
- **Extensibility**: We can extend the solution by adding new agents/tasks.
- **Interoperability**: We can integrate the solution with other systems by
  adding new agents/tasks.
- **Security/Privacy**: We can secure the solution by isolating sensitive agents/tasks.
- **Performance**: We can improve performance by isolating performance-critical agents/tasks.

## Task collaboration via sub-tasks

Langroid currently provides a mechanism for hierarchical (i.e. tree-structured)
task delegation: a `Task` object can add other `Task` objects
as sub-tasks, as shown in this pattern:

```py
from langroid import ChatAgent, ChatAgentConfig, Task

main_agent = ChatAgent(ChatAgentConfig(...))
main_task = Task(main_agent, ...)

helper_agent1 = ChatAgent(ChatAgentConfig(...))
helper_agent2 = ChatAgent(ChatAgentConfig(...))
helper_task1 = Task(agent1, ...)
helper_task2 = Task(agent2, ...)

main_task.add_sub_task([helper_task1, helper_task2])
```

What happens when we call `main_task.run()`?
Recall from the [previous section](chat-agent.md) that `Task.run()` works by
repeatedly calling `Task.step()` until `Task.done()` is True.
When the `Task` object has no sub-tasks, `Task.step()` simply tries
to get a valid response from the `Task`'s `ChatAgent`'s "native" responders,
in this sequence:
```py
[self.agent_response, self.llm_response, self.user_response] #(1)!
```

1. This is the default sequence in Langroid, but it can be changed by
   overriding [`ChatAgent.entity_responders()`][langroid.agent.base.Agent.entity_responders]

When a `Task` object has subtasks, the sequence of responders tried by
`Task.step()` consists of the above "native" responders, plus the
sequence of `Task.run()` calls on the sub-tasks, in the order in which
they were added to the `Task` object. For the example above, this means
that `main_task.step()` will seek a valid response in this sequence:

```py
[self.agent_response, self.llm_response, self.user_response, 
    helper_task1.run(), helper_task2.run()]
```
Fortunately, as noted in the [previous section](chat-agent.md),
`Task.run()` has the same type signature as that of the `ChatAgent`'s
"native" responders, so this works seamlessly. Of course, each of the
sub-tasks can have its own sub-tasks, and so on, recursively.
One way to think of this type of task delegation is that
`main_task()` "fails-over" to `helper_task1()` and `helper_task2()`
when it cannot respond to the current `pending_message` on its own.

## **Or Else** logic vs **And Then** logic
It is important to keep in mind how `step()` works: As each responder 
in the sequence is tried, when there is a valid response, the 
next call to `step()` _restarts its search_ at the beginning of the sequence
(with the only exception being that the human User is given a chance 
to respond after each non-human response). 
In this sense, the semantics of the responder sequence is similar to
**OR Else** logic, as opposed to **AND Then** logic.

If we want to have a sequence of sub-tasks that is more like
**AND Then** logic, we can achieve this by recursively adding subtasks.
In the above example suppose we wanted the `main_task` 
to trigger `helper_task1` and `helper_task2` in sequence,
then we could set it up like this:

```py
helper_task1.add_sub_task(helper_task2) #(1)!
main_task.add_sub_task(helper_task1)
```

1. When adding a single sub-task, we do not need to wrap it in a list.

## Next steps

In the [next section](two-agent-chat-num.md) we will see how this mechanism 
can be used to set up a simple collaboration between two agents.

