"""
Run like this:
python3 examples/experimental/mock-team.py
"""
from langroid.experimental.team import Team, Scheduler, OrElseScheduler, TaskComponent
import langroid as lr
from langroid.language_models.mock_lm import MockLMConfig
import logging

# Fix logging level type
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def sum_fn(s: str) -> str:
    """Dummy response for MockLM"""
    nums = [
        int(subpart)
        for part in s.split()
        for subpart in part.split(",")
        if subpart.isdigit()
    ]
    return str(sum(nums) + 1)


def make_task(name: str, sys: str = "") -> TaskComponent:
    llm_config = MockLMConfig(response_fn=sum_fn)
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            name=name,
        )
    )
    # set as single_round since there are no Tools
    task = lr.Task(agent, interactive=False, single_round=True)
    return TaskComponent(task)


if __name__ == "__main__":
    # Create task-components;
    # each task simply returns 1 + (sum of the numbers in the input)

    t1 = make_task("a1")
    t2 = make_task("a2")
    t3 = make_task("a3")

    # done conditions for each team
    def team1_done_condition(team: Team, scheduler: Scheduler) -> bool:
        return (
            scheduler.responder_counts.get("a1", 0) >= 2
            and scheduler.responder_counts.get("a2", 0) >= 2
        )

    def team2_done_condition(team: Team, scheduler: Scheduler) -> bool:
        return "a3" in scheduler.responders

    def general_team_done_condition(team: Team, scheduler: Scheduler) -> bool:
        # Example: all components have responded at least once
        return len(set(scheduler.responders)) == len(team.components)

    # Create teams
    team1 = Team("T1", done_condition=team1_done_condition)
    team2 = Team("T2", done_condition=team2_done_condition)

    team = Team("T", done_condition=general_team_done_condition)

    team1.add_scheduler(OrElseScheduler)
    team2.add_scheduler(OrElseScheduler)
    team.add_scheduler(OrElseScheduler)

    team.add([team1, team2])

    # Build hierarchy
    team1.add([t1, t2])
    team2.add(t3)

    # Set up listening
    team1.listen(team)  # gets input from parent team
    t1.listen(team1)  # gets input from parent team
    t2.listen(t1)  # gets input from sibling
    t1.listen(t2)  # gets input from sibling
    # TODO should we forbid listening to a component OUTSIDE the team?

    # t3 listens to its parent team2 =>
    # any input to team2 gets pushed to t3 when t3 runs
    team2.listen([t1, t2])  # team2 listens to internal components of team1
    t3.listen(team2)  # gets input from parent team2

    print("Running top-level team...")
    result = team.run("1")

    ##########
