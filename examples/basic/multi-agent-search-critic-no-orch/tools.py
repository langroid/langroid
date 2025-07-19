from typing import List

import typer

import langroid as lr

app = typer.Typer()


class QuestionTool(lr.ToolMessage):
    request: str = "question_tool"
    purpose: str = "Ask a SINGLE <question> that can be answered from a web search."
    question: str

    @classmethod
    def examples(cls) -> List[lr.ToolMessage]:
        return [
            cls(question="Which superconductor material was discovered in 2023?"),
            cls(question="What AI innovation did Meta achieve in 2024?"),
        ]


class AnswerTool(lr.ToolMessage):
    request: str = "answer_tool"
    purpose: str = "Present the <answer> to a web-search question"
    answer: str


class FinalAnswerTool(lr.ToolMessage):
    request: str = "final_answer_tool"
    purpose: str = """
        Present the intermediate <steps> and 
        final <answer> to the user's original <query>.
        """
    query: str
    steps: str
    answer: str

    @classmethod
    def examples(cls) -> List["lr.ToolMessage"]:
        return [
            (
                "I want to show my reasoning steps, along with my final answer",
                cls(
                    query="was Plato mortal?",
                    steps="1. Man is mortal. 2. Plato was a man.",
                    answer="Plato was mortal.",
                ),
            ),
            cls(
                query="Who was president during the moon landing?",
                steps="1. The moon landing was in 1969. 2. Kennedy was president "
                "during 1969.",
                answer="Kennedy was president during the moon landing.",
            ),
        ]


class FeedbackTool(lr.ToolMessage):
    request: str = "feedback_tool"
    purpose: str = """
    Provide <feedback> on the user's answer. If the answer is valid based on the
    reasoning steps, then the feedback MUST be EMPTY
    """
    feedback: str
    suggested_fix: str

    @classmethod
    def examples(cls) -> List["lr.ToolMessage"]:
        return [
            # just example
            cls(feedback="This looks fine!", suggested_fix=""),
            # thought + example
            (
                "I want to provide feedback on the reasoning steps and final answer",
                cls(
                    feedback="""
                    The answer is invalid because the conclusion does not follow from the
                    steps. Please check your reasoning and try again.
                    """,
                    suggested_fix="Check reasoning and try again",
                ),
            ),
        ]
