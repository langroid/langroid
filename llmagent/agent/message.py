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
    result: str  # result of agent method

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
        Return a string example of when the message should be used, possibly
        parameterized by the field values. This should be a valid english phrase for
        example,
        - "I want to know whether the file blah.txt is in the repo"
        - "What is the python version?"
        Returns:
            str: description of when the message should be used.
        """
        pass

    def not_use_when(self):
        """
        Return a string example of when the message should NOT be JSON formatted.
        This should be a valid 1st person phrase or question as in `use_when`.
        This method will be used to generate sample conversations of JSON-formatted
        questions, mixed in with questions that are not JSON-formatted.
        Unlike `use_when`, this method should not be parameterized by the field
        values, and also it should include THINKING and QUESTION lines.

        We supply default THINKING/QUESTION pairs, but subclasses can override these.
        Example:
            THINKING: I need to know the population of the US
            QUESTION: What is the population of the US?
        Returns:
            str: example of a situation when the message should NOT be JSON formatted.
        """
        return """
        THINKING: I need to know the population of the US
        QUESTION: What is the population of the US?
        """

    def usage_example(self):
        """
        Instruction to the LLM showing an example of how to use the message.
        Returns:
            str: description of how to use the message.
        """
        return f"""
        THINKING: {self.use_when()}        
        QUESTION: {self.json_example()}
        """

    def json_example(self):
        return self.json(indent=4, exclude={"result"})

    def sample_conversation(self, include_non_json=False):
        json_qa = f"""
        ExampleAssistant:
        {self.usage_example()}
        
        ExampleUser: {self.result}
        """

        if not include_non_json:
            return json_qa

        return (
            json_qa
            + f"""
        ExampleAssistant:
        {self.not_use_when()}
        
        ExampleUser: I don't know.
        """
        )
