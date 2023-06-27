from examples.dockerchat.message_validator_agent import MessageValidatorAgent
from llmagent.agent.chat_agent import ChatAgentConfig

cfg = ChatAgentConfig(vecdb=None, llm=None, name="Validator")

agent = MessageValidatorAgent(cfg)


# strictly validate expected format TO[xyz]:...
def test_validator():
    # various bad formats -> response
    response = agent.agent_response("TO:[Bob]:Hello Bob")
    assert response is not None
    response = agent.agent_response("TO:BOB: hi there")
    assert response is not None
    response = agent.agent_response("hello world")
    assert response is not None
    # correct format -> no response
    response = agent.agent_response("TO[BOB]: hi there")
    assert response is None
