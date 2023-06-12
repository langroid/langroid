"""
Structured messages to an agent, typically from an LLM, to be handled by
an agent. The messages could represent, for example:
- information or data given to the agent
- request for information or data from the agent
- request to run a method of the agent
"""

from abc import ABC, abstractmethod
from random import choice
from typing import Any, Dict, List

from pydantic import BaseModel

INSTRUCTION = """
    When one of these tools is applicable, you must express your request as
    "TOOL" followed by the request in JSON format. The fields will be based on the 
    tool description, which will be of the form:
    
    <tool_name>: description involving <arg1> maybe some other <arg2> and so on.
    
    The JSON format will be:
    {
        "request": "<tool_name>",
        "<arg1>": <value1>,
        "<arg2>": <value2>
    } 
    where it is important to note that <arg1> is the NAME of the argument, 
    and <value1> is the VALUE of the argument.
    
    For example suppose a tool with this description is available:
    
    country_capital: check if <city> is the capital of <country>.
    
    Now suppose you want to do this:
     
     "Check whether the capital of France is Paris",
      
    you realize that the `country_capital` tool is applicable, and so you must 
    ask in the following format: "TOOL: <JSON-formatted-request>", which in this 
    case will look like:

    TOOL: 
    \\{
        "request": "country_capital",
        "country": "France",
        "city": "Paris"
    \\}
    
    On the other hand suppose you want to:
    
    "Find out the population of France".
    
    In this case you realize there is no available TOOL for this, so you just ask in 
    natural language: "What is the population of France?"
    
    Whenever possible, AND ONLY IF APPLICABLE, use these TOOL, with the JSON syntax 
    specified above. When a TOOL is applicable, simply use this syntax, do not write 
    anything else. Only if no TOOL is exactly applicable, ask in natural language. 
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
        arbitrary_types_allowed = False
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

    def usage_example(self) -> str:
        """
        Instruction to the LLM showing an example of how to use the message.
        Returns:
            str: example of how to use the message
        """
        # pick a random example of the fields
        ex = choice(self.examples())
        return ex.json_example()

    def json_example(self) -> str:
        return self.json(indent=4, exclude={"result", "purpose"})

    def dict_example(self) -> Dict[str, Any]:
        return self.dict(exclude={"result", "purpose"})
