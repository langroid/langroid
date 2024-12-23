import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import langroid as lr

# Fix logging level type
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def user_message(msg: Union[str, lr.ChatDocument]) -> lr.ChatDocument:
    """Create a user-role msg from a string or ChatDocument"""
    if isinstance(msg, lr.ChatDocument):
        return msg
    return lr.ChatDocument(
        content=msg,
        metadata=lr.ChatDocMetaData(
            sender=lr.Entity.USER,
            sender_name="user",
        ),
    )


class InputContext:
    """Context for a Component to respond to"""

    def __init__(self) -> None:
        self.messages: List[lr.ChatDocument] = []

    def add(
        self, results: Union[str, List[str], lr.ChatDocument, List[lr.ChatDocument]]
    ) -> None:
        """
        Add messages to the input messages list
        """
        msgs: List[lr.ChatDocument] = []
        if isinstance(results, str):
            msgs = [user_message(results)]
        elif isinstance(results, lr.ChatDocument):
            msgs = [results]
        elif isinstance(results, list):
            if len(results) == 0:
                return
            if isinstance(results[0], str):
                msgs = [user_message(r) for r in results]
            else:
                msgs = [r for r in results if isinstance(r, lr.ChatDocument)]
        self.messages.extend(msgs)

    def clear(self) -> None:
        self.messages.clear()

    def get_context(self) -> lr.ChatDocument:
        """
        Construct a ChatDocument with sender = User, from the input messages
        accumulated so far.
        """
        if len(self.messages) == 0:
            return lr.ChatDocument(content="", metadata={"sender": lr.Entity.USER})
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
        self.responders: List[str] = []
        self.responder_counts: Dict[str, int] = {}
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
        self.name: str = ""

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

    def _notify(self, results: List[lr.ChatDocument]) -> None:
        logger.warning(f"{self.name} Notifying listeners...")
        for listener in self.listeners:
            logger.warning(f"--> Listener {listener.name} notified")
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
        results = []
        for comp in self.components:
            result = comp.run()
            if result:
                results.extend(result)
        self.current_result = results
        self.stepped = True

    def done(self) -> bool:
        """done after 1 step, i.e. all components have responded"""
        return self.stepped

    def result(self) -> List[lr.ChatDocument]:
        return self.current_result


class OrElseScheduler(Scheduler):
    """
    Implements "OrElse scheduling", i.e. if the components are A, B, C, then
    in each step, it will try for a valid response from A OrElse B OrElse C,
    i.e. the first component that gives a valid response is chosen.
    In the next step, it will start from the next component in the list,
    cycling back to the first component after the last component.
    (There may be a better name than OrElseScheduler though.)
    """

    def __init__(
        self,
        components: List[Component],
    ) -> None:
        super().__init__()
        self.components = components
        self.team: Optional[Team] = None
        self.current_index: int = 0

    def init_state(self) -> None:
        super().init_state()
        self.current_index = 0

    def is_valid(self, result: Optional[List[lr.ChatDocument]]) -> bool:
        return result is not None and len(result) > 0

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
        if self.team is None:
            return False
        return self.team.done(self)

    def result(self) -> List[lr.ChatDocument]:
        return self.current_result


class Team(Component):
    def __init__(
        self,
        name: str,
        done_condition: Optional[Callable[["Team", Scheduler], bool]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.components: List[Component] = []
        self.scheduler: Optional[Scheduler] = None
        self.done_condition = done_condition or Team.default_done_condition

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
        if hasattr(self.scheduler, "team"):
            setattr(self.scheduler, "team", self)

    def add(self, component: Union[Component, List[Component]]) -> None:
        if isinstance(component, list):
            self.components.extend(component)
        else:
            self.components.append(component)

    def reset(self) -> None:
        self.input.clear()
        if self.scheduler is not None:
            self.scheduler.init_state()

    def run(self, input: str | lr.ChatDocument | None = None) -> List[lr.ChatDocument]:
        if input is not None:
            self.input.add(input)
        if self.scheduler is None:
            raise ValueError(
                f"Team '{self.name}' has no scheduler. Call add_scheduler() first."
            )
        input_str = self.input.get_context().content
        logger.warning(f"Running team {self.name}... on input = {input_str}")
        # push the input of self to each component that's a listener of self.
        n_pushed = 0
        for comp in self.components:
            if comp in self.listeners:
                comp.input.add(self.input.messages)
                n_pushed += 1
        if len(self.input.messages) > 0 and n_pushed == 0:
            logger.warning(
                f"""
                Team {self.name} has input but no internal listeners,
                so this input will not be passed to any components,
                and you may not be able to run the team successfully.
                Make sure to set up components listening to parent team
                if needed.
                """
            )
        self.input.clear()

        result = self.scheduler.run()
        if len(result) > 0:
            self._notify(result)
        # clear own input since we've consumed it!
        self.input.clear()
        result_value = result[0].content if len(result) > 0 else "null"
        logger.warning(f"Team {self.name} done: {result_value}")
        return result


class TaskComponent(Component):
    def __init__(self, task: lr.Task) -> None:
        super().__init__()
        self.task = task
        self.name = task.agent.config.name

    def run(self, input: str | lr.ChatDocument | None = None) -> List[lr.ChatDocument]:
        if input is not None:
            self.input.add(input)
        input_msg = self.input.get_context()
        if input_msg.content == "":
            return []
        logger.warning(f"Running task {self.name} on input = {input_msg.content}")
        result = self.task.run(input_msg)
        result_value = result.content if result else "null"
        logger.warning(f"Task {self.name} done: {result_value}")
        result_list = [result] if result else []
        if len(result_list) > 0:
            self._notify(result_list)
            self.input.clear()  # clear own input since we just consumed it!
        return result_list
