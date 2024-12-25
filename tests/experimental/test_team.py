import langroid as lr
from langroid.experimental.team import OrElseScheduler, Scheduler, TaskNode, Team
from langroid.language_models.mock_lm import MockLMConfig


def make_task(name: str, response_fn):
    llm_config = MockLMConfig(response_fn=response_fn)
    agent = lr.ChatAgent(lr.ChatAgentConfig(llm=llm_config, name=name))
    task = lr.Task(agent, interactive=False, single_round=True)
    return TaskNode(task)


def test_basic_team():
    def sum_fn(s: str) -> str:
        nums = [int(n) for n in s.split() if n.isdigit()]
        return str(sum(nums) + 1)

    def done_condition(team: Team, scheduler: Scheduler) -> bool:
        return len(scheduler.responders) > 0

    t1 = make_task("a1", sum_fn)
    team = Team("test_team", done_condition=done_condition)
    team.add(t1)
    t1.listen(team)
    team.add_scheduler(OrElseScheduler)

    result = team.run("5")
    assert result[0].content == "6"


def test_team_with_listeners():
    def sum_fn(s: str) -> str:
        nums = [int(n) for n in s.split() if n.isdigit()]
        return str(sum(nums) + 1)

    def done_condition(team: Team, scheduler: Scheduler) -> bool:
        # done when each responder has run twice
        return len(scheduler.responders) == len(team.Nodes) * 2

    # Create tasks and teams
    t1 = make_task("a1", sum_fn)
    t2 = make_task("a2", sum_fn)
    team = Team("test_team", done_condition=done_condition)

    # Set up hierarchy and listening
    team.add([t1, t2])
    t1.listen(team)  # t1 listens to parent team
    t2.listen(t1)  # t2 listens to t1
    t1.listen(t2)  # t1 listens to t2
    team.add_scheduler(OrElseScheduler)

    result = team.run("5")
    # team -> push 5 -> t1
    # t1.run(5) -> 6 -> notify t2
    # t2.run(6) -> 7 -> notify t1
    # t1.run(7) -> 8 -> notify t2
    # t2.run(8) -> 9 -> done (since each Node has run twice)
    assert result[0].content == "9"
