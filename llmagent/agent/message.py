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
from random import choice


class ThoughtQuestionAnswer(BaseModel):
    """
    Represents a round in a conversation, with a
    thought & question from the assistant, and an answer from user (human or agent).
    """

    thought: str
    question: str
    answer: str

    def example(self) -> str:
        """
        Example to use in few-shot demos of JSON formatting instructions, or in
             sample conversations.
        Returns:
            string example of the message
        """
        return f"""
        THINKING: {self.thought}
        QUESTION: {self.question}
        """

    def conversation(self):
        """
        Example to use in a sample conversation between assistant and user.
        Returns:
            string example of the message
        """
        return f"""
        ExampleAssistant:
        {self.example()}
        
        ExampleUser:{self.answer}
        """


INSTRUCTION = """
    When one of these tools is applicable, you must express your request in 
    JSON format. The fields will be based on the tool description, which will be of 
    the form:
    
    <tool_name>: description involving <arg1> maybe some other <arg2> and so on.
    
    The JSON format will be:
    {
        "request": "<tool_name>",
        "<arg1>": <value1>,
        "<arg2>": <value2>
    } 
    where it is important to note that <arg1> is the NAME of the argument, 
    and <value1> is teh VALUE of the argument.
    
    For example if a tool has this description is available:
    
    country_capital: check if <city> is the capital of <country>,
    
    and you want to check whether the capital of France is Paris, you must the 
    ask in the following JSON format: 
    
    The JSON format will be:
    {
        "request": "country_capital",
        "country": "France",
        "city": "Paris"
    }
    
    But if you want to find out the population of France, you must ask in natural
    nautral language: "What is the population of France?"
    
    Whenever possible, AND ONLY IF APPLICABLE, use these tools, with the JSON syntax 
    specified above. When a tool is applicable, simply use this syntax, do not write 
    anything else. Only if no tool is exactly applicable, ask in natural language. 
    """


class AgentMessage(ABC, BaseModel):
    """
    A (structured) message to an agent, typically from an LLM, to be handled by
    the agent. The message could represent
    - information or data given to the agent
    - request for information or data from the agent
    Attributes:
        request (str): name of agent method to map to.
        purpose (str): purpose of agent method, expressed in general terms.
        result (str): result of agent method.
    """

    request: str
    purpose: str
    result: str

    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True

    @classmethod
    @abstractmethod
    def examples(cls) -> List["AgentMessage"]:
        """
        Examples to use in few-shot demos with JSON formatting instructions.
        Returns:
        """
        pass

    @abstractmethod
    def use_when(self):
        """
        Return a LIST of string examples (at least one, but ideally at least 2)
        of when the message should be used, possibly parameterized by the field
        values. This should be a valid english phrase for example,
        [ "I want to know which python version is needeed.",
          "I need to check the Python version."]

        The phrases should be in first person, and should be valid completions of
        "I will use this message when...".
        Returns:
            str: list of phrases showing when to use the message.
        """
        pass

    def non_usage_examples(self) -> List[ThoughtQuestionAnswer]:
        """
        Return a List of examples where the request should NOT be JSON formatted.
        This should be a valid 1st person phrase or question as in `use_when`.
        This method will be used to generate sample conversations of JSON-formatted
        questions, mixed in with questions that are not JSON-formatted.
        Unlike `use_when`, this method should not be parameterized by the field
        values, and also it should include THINKING and QUESTION lines.

        We supply defaults here, but subclasses can override these.
        It is important to adhere to the
        Returns:
            List of ThoughtRequestAnswer examples showing when
            the message should NOT be JSON formatted.
        """

        examples = [
            ThoughtQuestionAnswer(
                thought="I need to know the population of the US",
                question="What is the population of the US?",
                answer="328,239,523",
            ),
            ThoughtQuestionAnswer(
                thought="I want to check how many files are in the repo",
                question="How many files are in the repo?",
                answer="1,000",
            ),
        ]
        return examples

    def non_usage_example(self, conversation: bool = True) -> str:
        if conversation:
            return choice(self.non_usage_examples()).conversation()
        else:
            return choice(self.non_usage_examples()).example()

    def usage_example(self, conversation: bool = False) -> str:
        """
        Instruction to the LLM showing an example of how to use the message.
        Args:
            conversation (bool): whether the example is to be used in a sample
            conversation, in which case we should include roles (Assistant, User) and
            sample answer.
        Returns:
            str: description of how to use the message, or a sample conversation.
        """
        # pick a random example of the fields
        ex = choice(self.examples())
        return ex.json_example()

    def json_example(self):
        return self.json(indent=4, exclude={"result", "purpose", "instruction"})

    def dict_example(self):
        return self.dict(exclude={"result", "purpose", "instruction"})

    def sample_conversation(self, json_only=True):
        """
        Generate a sample conversation with the message, possibly including non-JSON
            formatted questions.
        Args:
            json_only: whether to only include JSON formatted example
        Returns:

        """
        json_qa = self.usage_example(conversation=True)
        if json_only:
            return json_qa
        return json_qa + "\n\n" + self.non_usage_example(conversation=True)
