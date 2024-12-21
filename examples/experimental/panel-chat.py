from langroid.experimental.team import Team, Scheduler, OrElseScheduler, TaskComponent
import langroid as lr
import langroid.language_models as lm
from langroid.language_models.openai_gpt import OpenAIGPTConfig
import logging

# Fix logging level type
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

AGENTS = ["A", "B", "C"]


def make_task(name: str) -> TaskComponent:
    llm_config = OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4o,
    )
    other_agents = [a for a in AGENTS if a != name]
    agent1 = other_agents[0]
    agent2 = other_agents[1]
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            name=name,
            system_message=f"""
            You are playing a game with two other agents ({agent1} and {agent2}).
            In each round, each of you must choose an ARBITRARY number between 1 and 10,
            that has NOT YET been chosen by YOU OR any of the OTHER agents.
            YOU DO NOT NEED TO CHOOSE numbers sequentially.
            
            If you CAN choose such a number, simply say it and NOTHING ELSE.
            
            If you CANNOT choose a number, you must say "STUCK" and 
            show ALL the choices of the ALL agents in this format:
            
            AgentName1: 3, 5, 7
            AgentName2: 2, 4, 6
            SELF: ...
            ... and so on.
            """,
        )
    )
    # set as single_round since there are no Tools
    # NOTE - we set restart=False so state is retained between runs
    task = lr.Task(agent, interactive=False, single_round=True, restart=False)
    return TaskComponent(task)


if __name__ == "__main__":
    # Create task-components
    ta = make_task("A")
    tb = make_task("B")
    tc = make_task("C")

    def team_done_condition(team: Team, scheduler: Scheduler) -> bool:
        return any("STUCK" in r.content for r in scheduler.current_result)

    # Create teams
    team = Team("T", done_condition=team_done_condition)

    team.add_scheduler(OrElseScheduler)

    team.add([ta, tb, tc])

    # Set up listening
    # team2.listen(team1)  # listens to team1 final result
    ta.listen(team)
    ta.listen([tb, tc])
    tb.listen([ta, tc])
    tc.listen([ta, tb])

    print("Running top-level team...")
    result = team.run("get started!")

    ##########
