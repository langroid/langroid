"""
Structured messages to an agent, typically from an LLM, to be handled by
an agent. The messages could represent, for example:
- information or data given to the agent
- request for information or data from the agent
- request to run a method of the agent
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List


class AgentMessage(ABC, BaseModel):
    """
    A (structured) message to an agent, typically from an LLM, to be handled by
    the agent. The message could represent
    - information or data given to the agent
    - request for information or data from the agent
    """

    request: str  # name of agent method to map to

    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Generate a list of instances of the subclass with example field values.
        If the fields have different numbers of examples, the minimum number
        of examples is used for all fields.

        Returns:
            List[AgentMessage]: List of instances of the subclass.
        """
        examples = {}
        min_num_examples = 1000
        for field_name, field in cls.__fields__.items():
            if field_name == "request":
                continue
            field_examples = field.field_info.extra.get("examples")
            if field_examples:
                examples[field_name] = field_examples
                min_num_examples = min(min_num_examples, len(field_examples))

        if not examples or min_num_examples == 0:
            return []

        field_names = examples.keys()
        combined_examples = [
            dict(zip(field_names, example_values[:min_num_examples]))
            for example_values in zip(*examples.values())
        ]
        instance_list = [cls(**example) for example in combined_examples]

        return instance_list

    @abstractmethod
    def use_when(self):
        """
        Return a string describing when the message should be used, possibly
        parameterized by the field values. This should be a valid english phrase for
        example,
        - "To check whether the number 3 is smaller than 4", or
        - "When you want to check whether file foo.txt exists"
        The returned phrase P should be such that the extended phrase
        "{P}, write the JSON string: ..." is a valid instruction for the LLM.
        Returns:
            str: description of when the message should be used.
        """
        pass

    def usage_instruction(self):
        """
        Instruction to the LLM showing an example of how to use the message.
        Returns:
            str: description of how to use the message.
        """
        return f"""
        {self.use_when()}, write the JSON string:        
        {self.json_example()}
        """

    def json_example(self):
        return self.json(indent=4)

    def sample_conversation(self):
        # todo need to think about whether to allow a "result" field.
        pass
        # return f"""
        # ExampleAssistant: {self.json_example()}
        #
        # ExampleUser: {self.result}
        # """
