from langroid.utils.configuration import Settings, set_global, settings


def test_update_global_settings():
    """
    Test that we can dynamically update the global settings object.
    """
    set_global(Settings(debug=True))
    assert settings.debug is True

    set_global(Settings(debug=False))
    assert settings.debug is False
