import langroid as lr
from langroid.agent.batch import run_batch_tasks
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.mock_lm import MockLMConfig


class DummyTool(ToolMessage):
    request: str = "dummy"
    purpose: str = "to show a dummy tool"

    value: int


def make_typed_dummy_task():
    # MockLM always returns a valid DummyTool JSON payload
    mock_json = DummyTool(value=42).json()
    cfg = ChatAgentConfig(
        name="DummyAgent",
        llm=MockLMConfig(default_response=mock_json),
        handle_llm_no_tool=f"Please use {DummyTool.name()}",
        system_message=f"Always return {DummyTool.name()} with a value.",
    )
    agent = ChatAgent(cfg)
    agent.enable_message(DummyTool)
    task_cfg = lr.TaskConfig(done_if_tool=True)
    # Typed Task: expect single-run to return DummyTool
    task = lr.Task(agent, interactive=False, config=task_cfg)[DummyTool]
    return agent, task


def test_single_run_typed_task_returns_dummy_tool():
    agent, task = make_typed_dummy_task()
    result = task.run("any input")
    assert isinstance(result, DummyTool)
    assert result.value == 42

    task_clone = task.clone(1)
    result2 = task_clone.run("any input")
    assert isinstance(result2, DummyTool)


def test_batched_typed_task_returns_typed_objects():
    """
    This intentionally asserts the behavior we WANT (typed results from batch),
    """
    agent, task = make_typed_dummy_task()

    inputs = ["a", "b"]
    results = run_batch_tasks(
        task,
        inputs,
        input_map=lambda x: x,
        output_map=lambda x: x,  # identity; we expect typed but get ChatDocument
        batch_size=2,
        sequential=True,
        turns=1,
    )

    # What we would like to be true (but isn't):
    assert all(isinstance(r, DummyTool) for r in results)
