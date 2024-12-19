from abc import ABC, abstractmethod
from typing import Any, List

import langroid as lr
from langroid import ChatDocument


class InputContext:
    def __init__(self):
        self.messages: List[lr.ChatDocument] = []

    def add(self, results: List[lr.ChatDocument]):
        self.messages.extend(results)

    def clear(self):
        self.messages.clear()


class Scheduler(ABC):
    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def result(self) -> lr.ChatDocument:
        pass

    def run(self):
        while not self.done():
            self.step()
        return self.result()


class Component(ABC):
    def __init__(self):
        self.input = InputContext()

    @abstractmethod
    def run(self, inputs: List[Any]) -> Any:
        pass

    def listen(self, component: Component | List[Component]):
        if isinstance(component, list):
            for comp in component:
                comp.listeners.append(self)
        else:
            component.listeners.append(self)

    @property
    def listeners(self) -> List["Component"]:
        if not hasattr(self, "_listeners"):
            self._listeners = []
        return self._listeners


class SimpleScheduler(Scheduler):
    def __init__(
        self, components: List[Component], inputs: List[lr.ChatDocument] = None
    ):
        self.components: List[Component] = components
        self.inputs: List[lr.ChatDocument] = inputs or []
        self.current_results: List[lr.ChatDocument] = []
        self.stepped: bool = False

    def step(self):
        self.current_results = [comp.run(self.inputs) for comp in self.components]
        self.stepped = True

    def done(self):
        return self.stepped  # Now returns True only after stepping

    def result(self):
        return self.current_results


class OrElseScheduler(Scheduler):
    def __init__(self, components, inputs=None):
        self.components = components
        self.inputs = inputs or []
        self.current_result = None
        self.done_flag = False

    def is_valid(self, result) -> bool:
        return result is not None and result != ""

    def step(self):
        for comp in self.components:
            result = comp.run(comp.input.messages)
            if self.is_valid(result):
                self.current_result = result
                self.done_flag = True
                return
        self.done_flag = True

    def done(self):
        return self.done_flag

    def result(self):
        return self.current_result


class Team(Component):
    def __init__(self, name: str, scheduler_class=SimpleScheduler):
        super().__init__()
        self.name = name
        self.components = []
        self.scheduler_class = scheduler_class

    def add(self, component: Union[Component, List[Component]]):
        if isinstance(component, list):
            self.components.extend(component)
        else:
            self.components.append(component)

    def listen(self, team: "Team"):
        # TODO Can a team listen to a component outside of itself?
        team.listeners.append(self)

    def run(self, inputs=None) -> Any:
        all_inputs = inputs or self.input.messages
        scheduler = self.scheduler_class(self.components, inputs=all_inputs)
        result = scheduler.run()
        self._notify(result)
        self.input.clear()
        return result

    def _notify(self, results: List[Any]):
        for listener in self.listeners:
            listener.input.add(results)


# Example of existing agent class
class DummyAgent:
    def __init__(self, name):
        self.name = name

    def process(self, data):
        return f"{self.name} processed: {data}"


# Adapter for existing agent
class AgentAdapter(Component):
    def __init__(self, agent: DummyAgent):
        super().__init__()
        self.agent = agent

    def run(self, inputs: List[Any]) -> str:
        input_str = ", ".join(str(x) for x in inputs) if inputs else "no input"
        return self.agent.process(input_str)


if __name__ == "__main__":
    # Create agents
    agent1 = AgentAdapter(DummyAgent("Agent1"))
    agent2 = AgentAdapter(DummyAgent("Agent2"))
    agent3 = AgentAdapter(DummyAgent("Agent3"))

    # Create teams
    team1 = Team("Team1")
    team2 = Team("Team2")

    # Build hierarchy
    team1.add([agent1, agent2])
    team2.add(agent3)

    # Set up listening
    team2.listen(team1)
    agent1.listen(agent2)
    agent2.listen(agent1)

    # TODO - who will orchestrate team1 + team2 ?
    # Run scenarios
    print("Running team1...")
    result1 = team1.run(["Start discussion"])
    print(f"Team1 result: {result1}")
    print(f"Agent1's inputs: {agent1.input.messages}")
    print(f"Agent2's inputs: {agent2.input.messages}")

    print("\nRunning team2...")
    result2 = team2.run()
    print(f"Agent3's inputs from agent1: {agent3.input.messages}")
    print(f"Team2 result: {result2}")
