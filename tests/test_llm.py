from llmagent.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
from llmagent.language_models.base import LLMMessage, Role


def test_openai_gpt():
    cfg = OpenAIGPTConfig(
        type="openai",
        max_tokens=100,
        chat_model="gpt-3.5-turbo",
        completion_model="text-davinci-003",
    )

    mdl = OpenAIGPT(config=cfg)

    # completion mode
    question = "What is the capital of france?"

    response = mdl.generate(prompt=question, max_tokens=10)
    assert "Paris" in response.message

    # chat mode
    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assitant"),
        LLMMessage(role=Role.USER, content=question),
    ]
    response = mdl.chat(messages=messages, max_tokens=10)
    assert "Paris" in response.message
