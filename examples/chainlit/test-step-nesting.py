"""
Test whether the current chainlit version shows nested steps as expected.
Note that this does NOT show what you'd expect, due to breaking changes in Chainlit.

Two things to look for:
(1) are all types of steps shown, or only type = "tool"?
(2) when step B has parent_id pointing to Step A, we want to see Step B shown:
    - nested under Step A
    - shown in a chronologically correct order, i.e. if Step A says "hello",
        then calls Step B, then step B should be shown AFTER the "hello" message from A.

(1) is fine in chainlit 1.1.202, i.e. all steps are shown whether tools or not
    but in 1.1.300, only type = "tool" steps are shown.
    For example if the `type` params are other than "tool" in the example below,
    the steps will not show up in the chat.
(2) is broken in 1.1.202 -- the sub-step is correctly nested BUT always shows up
    at the TOP, and can look very unintuitive, as this example shows.
"""

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    a_step = cl.Step(name="A", type="tool")
    a_step.output = "asking B"
    await a_step.send()

    b_step = cl.Step(
        name="B",
        parent_id=a_step.id,
        type="tool",
    )
    b_step.output = "asking C"
    await b_step.send()

    c_step = cl.Step(
        name="C",
        parent_id=b_step.id,
        type="tool",
    )
    c_step.output = "C answered!"
    await c_step.send()
