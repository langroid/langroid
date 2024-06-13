from langroid.utils.globals import GlobalState


class _TestGlobals(GlobalState):
    """Test-specific global variables.
    (This is how users should define their own global variables)
    """

    some_variable: int = 0
    another_variable: str = ""
    mapping: dict = {}


def test_initial_global_state():
    """
    Test that the global state initializes with the default values.
    """
    assert _TestGlobals.get_value("some_variable") == 0
    assert _TestGlobals.get_value("another_variable") == ""
    assert _TestGlobals.get_value("mapping") == {}


def test_set_global_state():
    """
    Test setting new values on the global state.
    """
    _TestGlobals.set_values(some_variable=5, another_variable="Test")

    assert _TestGlobals.get_value("some_variable") == 5
    assert _TestGlobals.get_value("another_variable") == "Test"

    _TestGlobals.set_values(some_variable=7, another_variable="hello")

    assert _TestGlobals.get_value("some_variable") == 7
    assert _TestGlobals.get_value("another_variable") == "hello"

    _TestGlobals.set_values(mapping={"k1": "v1", "k2": "v2"})

    assert _TestGlobals.get_value("mapping")["k1"] == "v1"
    assert _TestGlobals.get_value("mapping")["k2"] == "v2"


def test_singleton_behavior():
    """
    Test that the global state behaves as a singleton.
    """
    first_instance = _TestGlobals.get_instance()
    second_instance = _TestGlobals.get_instance()

    assert first_instance is second_instance

    # Modify using one instance and check with the other
    first_instance.set_values(some_variable=10)
    assert second_instance.get_value("some_variable") == 10

    first_instance.set_values(mapping={"k1": "v1", "k2": "v2"})
    assert second_instance.get_value("mapping")["k1"] == "v1"
    assert second_instance.get_value("mapping")["k2"] == "v2"
