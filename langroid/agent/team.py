import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union

import langroid as lr
from langroid.language_models.mock_lm import MockLMConfig

logging.basicConfig(level=logging.warning)
logger = logging.getLogger(__name__)


def sum_fn(s: str) -> str:
    nums = [
        int(subpart)
        for part in s.split()
        for subpart in part.split(",")
        if subpart.isdigit()
    ]
    return str(sum(nums) + 1)


def user_message(s: str) -> lr.ChatDocument:
    return lr.ChatDocument(
        content=s,
        metadata=lr.ChatDocMetaData(
            sender=lr.Entity.USER,
            sender_name="user",
        ),
    )


class InputContext:
    """Context for a Team to respond to"""

    def __init__(self) -> None:
        self.messages: List[lr.ChatDocument] = []

    def add(
        self, results: str | List[str] | lr.ChatDocument | List[lr.ChatDocument]
    ) -> None:
        """
        Add messages to the input messages list
        """
        msgs = []
        if isinstance(results, str):
            msgs = [user_message(results)]
        elif isinstance(results, lr.ChatDocument):
            msgs = [results]
        elif isinstance(results, list):
            if isinstance(results[0], str):
                msgs = [user_message(r) for r in results]
            else:
                msgs = results
        self.messages.extend(msgs)

    def clear(self) -> None:
        self.messages.clear()

    def get_context(self) -> lr.ChatDocument:
        content = "\n".join(
            f"{msg.metadata.sender_name}: {msg.content}" for msg in self.messages
        )
        return lr.ChatDocument(content=content, metadata={"sender": lr.Entity.USER})


class Scheduler(ABC):
    """Schedule the Components of a Team"""

    def __init__(self) -> None:
        self.init_state()

    def init_state(self) -> None:
        self.stepped = False
        self.responders = []
        self.responder_counts = {}
        self.current_result: List[lr.ChatDocument] = []

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def result(self) -> List[lr.ChatDocument]:
        pass

    def run(self) -> List[lr.ChatDocument]:
        self.init_state()
        while not self.done():
            self.step()
        return self.result()


class Component(ABC):
    """A component of a Team"""

    def __init__(self) -> None:
        self.input = InputContext()
        self._listeners: List["Component"] = []

    @abstractmethod
    def run(self) -> List[lr.ChatDocument]:
        pass

    def listen(self, component: Union["Component", List["Component"]]) -> None:
        if isinstance(component, list):
            for comp in component:
                comp.listeners.append(self)
        else:
            component.listeners.append(self)

    @property
    def listeners(self) -> List["Component"]:
        return self._listeners

    def _notify(self, results: List[Any]) -> None:
        for listener in self.listeners:
            listener.input.add(results)


class SimpleScheduler(Scheduler):
    def __init__(
        self,
        components: List[Component],
    ) -> None:
        super().__init__()
        self.components = components  # Get components from team
        self.stepped: bool = False

    def step(self) -> None:
        self.current_result = [
            comp.run(comp.input.messages) for comp in self.components
        ]
        self.current_result = [r for r in self.current_result if r is not None]
        self.stepped = True

    def done(self) -> bool:
        """done after 1 step, i.e. all components have responded"""
        return self.stepped

    def result(self) -> List[lr.ChatDocument]:
        return self.current_result


class OrElseScheduler(Scheduler):
    def __init__(
        self,
        components: List[Component],
    ) -> None:
        super().__init__()
        self.components = components
        self.team = None

    def init_state(self) -> None:
        super().init_state()
        self.current_index = 0

    def is_valid(self, result: Any) -> bool:
        return result is not None and result != ""

    def step(self) -> None:
        start_index = self.current_index
        n = len(self.components)

        for i in range(n):
            idx = (start_index + i) % n
            comp = self.components[idx]
            result = comp.run()
            if self.is_valid(result):
                self.responders.append(comp.name)
                self.responder_counts[comp.name] = (
                    self.responder_counts.get(comp.name, 0) + 1
                )
                self.current_result = result
                # cycle to next component
                self.current_index = (idx + 1) % n
                return

    def done(self) -> bool:
        return self.team.done(self)

    def result(self) -> List[lr.ChatDocument]:
        return self.current_result


class Team(Component):
    def __init__(
        self, name: str, done_condition: Callable[["Team", Scheduler], bool] = None
    ) -> None:
        super().__init__()
        self.name = name
        self.components: List[Component] = []
        self.scheduler = None
        self.done_condition = done_condition or self.default_done_condition

    def set_done_condition(
        self, done_condition: Callable[["Team", Scheduler], bool]
    ) -> None:
        self.done_condition = done_condition

    def done(self, scheduler: Scheduler) -> bool:
        return self.done_condition(self, scheduler)

    def default_done_condition(self, scheduler: Scheduler) -> bool:
        # Default condition, can be overridden
        return False

    def add_scheduler(self, scheduler_class: type) -> None:
        self.scheduler = scheduler_class(self.components)
        self.scheduler.team = self

    def add(self, component: Union[Component, List[Component]]) -> None:
        if isinstance(component, list):
            self.components.extend(component)
        else:
            self.components.append(component)

    def listen(self, component: Union[Component, List[Component]]) -> None:
        if isinstance(component, list):
            for comp in component:
                comp.listeners.append(self)
        else:
            component.listeners.append(self)

    def run(self) -> List[lr.ChatDocument]:
        if self.scheduler is None:
            raise ValueError(
                f"Team '{self.name}' has no scheduler. Call add_scheduler() first."
            )
        logger.warning(f"Running team {self.name}...")
        result = self.scheduler.run()
        if len(result) > 0:
            self._notify(result)
        # clear own input since we've consumed it!
        self.input.clear()
        result_value = result[0].content if len(result) > 0 else "null"
        logger.warning(f"Team {self.name} done: {result_value}")
        return result


class DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, data: str) -> str:
        return f"{self.name} processed: {data}"


class TaskComponent(Component):
    def __init__(self, task: lr.Task) -> None:
        super().__init__()
        self.task = task
        self.name = task.agent.config.name

    def run(self) -> List[lr.ChatDocument]:
        logger.warning(f"Running task {self.name}...")
        input = self.input.get_context()
        result = self.task.run(input)
        result_value = result.content if result else "null"
        logger.warning(f"Task {self.name} done: {result_value}")
        if result is not None:
            self._notify(result if isinstance(result, list) else [result])
        self.input.clear()
        return [result]


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
    # Create agents, tasks
    t1 = make_task("a1")
    t2 = make_task("a2")
    t3 = make_task("a3")

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
    team1 = Team("Team1", done_condition=team1_done_condition)
    team2 = Team("Team2", done_condition=team2_done_condition)

    team = Team("Team", done_condition=general_team_done_condition)

    team1.add_scheduler(OrElseScheduler)
    team2.add_scheduler(OrElseScheduler)
    team.add_scheduler(OrElseScheduler)

    team.add([team1, team2])

    # Build hierarchy
    team1.add([t1, t2])
    team2.add(t3)

    # Set up listening
    team2.listen(team1)  # listens to team1 final result
    t2.listen(t1)
    t1.listen(t2)
    t3.listen([t1, t2])

    # TODO - we should either define which component of a team gets the teams inputs,
    # or explicitly add messages to a specific component of the team

    t1.input.add("1")

    # Run scenarios
    print("Running top-level team...")
    result = team.run()
