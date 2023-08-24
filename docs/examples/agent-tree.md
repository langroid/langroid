# Hierarchical computation with Langroid Agents 

Here is a simple example showing tree-structured computation
where each node in the tree is handled by a separate agent.
This is a toy numerical example, and illustrates:

- how to have agents organized in a hiearchical structure to accomplish a task 
- the use of global state accessible to all agents, and 
- the use of tools/function-calling.

## The Task

The task consists of performing this calculation for a given input number n:

```python
def Main(n):
    if n is odd:
        return (3*n+1) + n
    else:
        if n is divisible by 10:
            return n/10 + n
        else:
            return n/2 + n
```

## Using function composition

Imagine we want to do this calculation using a few auxiliary functions:

```python
def Main(n):
    # return non-null value computed by Odd or Even
    return Odd(n) or Even(n)

def Odd(n):
    # Handle odd n
    if n is odd:
        new = 3*n+1
        return Adder(new)
    else:
        return None
    
def Even(n):
    # Handle even n: return non-null value computed by EvenZ or EvenNZ
    return EvenZ(n) or EvenNZ(n)

def EvenZ(n):
    # Handle even n divisible by 10, i.e. ending in Zero
    if n is divisible by 10:
        new = n/10
        return Adder(new)
    else:
        return None
    
def EvenNZ(n):
    # Handle even n not divisible by 10, i.e. not ending in Zero
    if n is not divisible by 10:
        new = n/2
        return Adder(new)
    else:
        return None  

def Adder(new):
    # Add new to starting number, available as global variable n
    return new + n
```

## Mapping to a tree structure

This compositional/nested computation can be represented as a tree:

```plaintext
Main
|-- Even
|   |-- EvenZ
|   |   `-- Adder
|   |-- EvenNZ
|       `-- Adder
`-- Odd
    `-- Adder
```

Let us specify the behavior we would like for each node, in a 
"decoupled" way, i.e. we don't want a node to be aware of the other nodes.
As we see later, this decoupled design maps very well onto Langroid's
multi-agent task orchestration. To completely define the node behavior,
we need to specify how it handles an "incoming" number $n$ (from a parent node 
or user), and how it handles a "result" number $r$ (from a child node).

- `Main`: 
    - incoming $n$: simply send down $n$, record the starting number $n_0 = n$ as a global variable. 
    - result $r$: return $r$.
- `Odd`: 
    - incoming $n$: if n is odd, send down $3*n+1$, else return None
    - result $r$: return $r$
- `Even`: 
    - incoming $n$: if n is even, send down $n$, else return None
    - result $r$: return $r$
- `EvenZ`: (guaranteed by the tree hierarchy, to receive an even number.)  
    - incoming $n$: if n is divisible by 10, send down $n/10$, else return None
    - result $r$: return $r$
- `EvenNZ`: (guaranteed by the tree hierarchy, to receive an even number.)
    - incoming $n$: if n is not divisible by 10, send down $n/2$, else return None
    - result $r$: return $r$
- `Adder`:
    - incoming $n$: return $n + n_0$ where $n_0$ is the 
    starting number recorded by Main as a global variable.
    - result $r$: Not applicable since `Adder` is a leaf node.
  
## From tree nodes to Langroid Agents 

Let us see how we can perform this calculation using multiple Langroid agents, where

- we define an agent corresponding to each of the nodes above, namely `Main, Odd, EvenZ, EvenNZ, Adder`
- we wrap each Agent into a Task, and use the `Task.add_subtask()` method to connect the agents into 
  the above tree structure.

Below is one way to do this using Langroid. We designed this with the following
desirable features:

- Decoupling: Each agent is instructed separately, without mention of any other agents
  (E.g. Even agent does not know about Odd Agent, or EvenZ agent, etc).
  In particular, this means agents will not be "addressing" their message
  to specific other agents, e.g. send number to Odd agent when number is odd,
  etc. Allowing addressing would make the solution easier to implement,
  but would not be a decoupled solution.
  Instead, we want Agents to simply put the number "out there", and have it handled
  by an applicable agent, in the task loop (which consists of the agent's responders,
  plus any sub-task `run` methods).

- Simplicity: Keep the agent instructions relatively simple. We would not want a solution
  where we have to instruct the agents (their LLMs) in convoluted ways. 

One way naive solutions fail is because agents are not able to distinguish between
a number that is being "sent down" the tree as input, and a number that is being
"sent up" the tree as a result from a child node.

We use a simple trick: we mark returned values using the RESULT keyword,
and instruct the LLMs to generate and recognize these tags. In addition,
we leverage some features of Langroid's task orchestration:

- When `llm_delegate` is `True`, if the LLM says `DONE [rest of msg]`, the task is
  considered done, and the result of the task is `[rest of msg]` (i.e the part after `DONE`).
- In the task loop's `step()` function (which seeks a valid message during a turn of
  the conversation) when any responder says `DO-NOT-KNOW`, it is not considered a valid
  message, and the search continues to other responders, in round-robin fashion.

See the [`chat-tree.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-tree.py) 
example for an implementation of this solution. You can run that example as follows:
```bash
python3 examples/basic/chat-tree.py
```
