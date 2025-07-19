from typing import Optional
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.language_models.base import LLMMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.utils.object_registry import ObjectRegistry

register_object = ObjectRegistry.register_object


class A(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    my_b_id: Optional[str] = None
    parent_id: Optional[str] = None
    child_id: Optional[str] = None

    def my_b(self) -> Optional["B"]:
        return ObjectRegistry.get(self.my_b_id) if self.my_b_id else None

    @property
    def parent(self) -> Optional["A"]:
        return ObjectRegistry.get(self.parent_id) if self.parent_id else None

    @property
    def child(self) -> Optional["A"]:
        return ObjectRegistry.get(self.child_id) if self.child_id else None


class B(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    my_a_id: Optional[str] = None

    def my_a(self) -> Optional["A"]:
        return ObjectRegistry.get(self.my_a_id) if self.my_a_id else None


@pytest.fixture
def create_objects():
    """Fixture to create and register instances of A and B."""
    a1 = A()
    register_object(a1)
    b1 = B(my_a_id=a1.id)
    register_object(b1)
    a1.my_b_id = b1.id
    a2 = A(parent_id=a1.id)
    register_object(a2)
    a1.child_id = a2.id
    return a1, a2, b1


def test_id_creation(create_objects):
    """Test if objects have valid UUIDs as IDs."""
    a1, a2, b1 = create_objects
    assert len(a1.id) == 36, "A1 ID should be a valid UUID"
    assert len(a2.id) == 36, "A2 ID should be a valid UUID"
    assert len(b1.id) == 36, "B1 ID should be a valid UUID"


def test_object_lookup(create_objects):
    """Test if objects can be retrieved correctly using their IDs."""
    a1, a2, b1 = create_objects
    assert ObjectRegistry.get(a1.id) is a1, "Lookup for A1 should return A1"
    assert ObjectRegistry.get(a2.id) is a2, "Lookup for A2 should return A2"
    assert ObjectRegistry.get(b1.id) is b1, "Lookup for B1 should return B1"


def test_a_to_a_links(create_objects):
    """Test parent and child links between instances of A."""
    a1, a2, _ = create_objects
    assert a2.parent is a1, "A2's parent should be A1"
    assert a1.child is a2, "A1's child should be A2"


def test_a_b_links(create_objects):
    """Test links between instances of A and B."""
    a1, _, b1 = create_objects
    assert b1.my_a() is a1, "B1's my_a should point to A1"
    assert a1.my_b() is b1, "A1's my_b should point to B1"


def test_remove_object(create_objects):
    """Test the removal of an object from the registry."""
    a1, a2, b1 = create_objects
    # Ensure the object is initially in the registry
    assert ObjectRegistry.get(a1.id) is not None
    # Remove the object
    ObjectRegistry.remove(a1.id)
    # Ensure the object is no longer in the registry
    assert ObjectRegistry.get(a1.id) is None


def test_cleanup_registry(create_objects):
    """Test the cleanup of the registry to remove None references."""
    a1, a2, b1 = create_objects
    # Introduce a None entry manually for testing
    ObjectRegistry.registry["dummy_id"] = None
    # Ensure "dummy_id" is in the registry before cleanup
    assert "dummy_id" in ObjectRegistry.registry
    # Perform cleanup
    ObjectRegistry.cleanup()
    # "dummy_id" should be removed post cleanup
    assert "dummy_id" not in ObjectRegistry.registry
    # Ensure other objects are still in the registry
    assert ObjectRegistry.get(a1.id) is not None
    assert ObjectRegistry.get(a2.id) is not None
    assert ObjectRegistry.get(b1.id) is not None


def test_chat_documents():
    # ChatDocument instances are automatically registered in the ObjectRegistry
    a_doc = ChatDocument(content="astuff", metadata=ChatDocMetaData(sender=Entity.LLM))
    b_doc = ChatDocument(content="bstuff", metadata=ChatDocMetaData(sender=Entity.LLM))

    a_doc.metadata.parent_id = b_doc.id()
    b_doc.metadata.child_id = a_doc.id()

    assert ChatDocument.from_id(a_doc.id()) is a_doc
    assert ChatDocument.from_id(b_doc.id()) is b_doc

    assert ObjectRegistry.get(a_doc.id()) is a_doc, "Lookup for A should return A"
    assert ObjectRegistry.get(b_doc.id()) is b_doc, "Lookup for B should return B"

    assert a_doc.parent is b_doc, "A's parent should be B"
    assert b_doc.child is a_doc, "B's child should be A"

    # convert to LLMMessage
    llm_msg = ChatDocument.to_LLMMessage(a_doc)[0]
    assert isinstance(llm_msg, LLMMessage)
    assert llm_msg.chat_document_id == a_doc.id()


def test_agent_chat_document_link():
    agent = ChatAgent(
        ChatAgentConfig(llm=MockLMConfig(default_response="7"))
    )  # auto-registered
    agent.message_history = [
        LLMMessage(role="system", content="You are helpful"),
        LLMMessage(role="user", content="hello"),
        LLMMessage(role="assistant", content="hi there"),
    ]
    response_doc = agent.llm_response("3+4?")
    assert response_doc is not None
    assert isinstance(response_doc, ChatDocument)
    assert response_doc.metadata.agent_id == agent.id
    assert response_doc.metadata.msg_idx == 4
    assert ObjectRegistry.get(response_doc.id()) is response_doc
    assert ObjectRegistry.get(agent.id) is agent
    last_msg = agent.message_history[-1]
    assert (
        last_msg.chat_document_id == response_doc.id()
    ), "Last message (LLM response) should be linked to the response chat document"

    assert (
        ObjectRegistry.get(last_msg.chat_document_id) is response_doc
    ), "Lookup from last message should return the response chat document"
