from langroid.utils.configuration import quiet_mode, settings
from langroid.utils.output import status


def test_quiet_mode():
    assert not settings.quiet

    with quiet_mode():
        assert settings.quiet

    assert not settings.quiet


def test_nested_quiet_mode():
    assert not settings.quiet

    with quiet_mode():
        assert settings.quiet

        with quiet_mode(quiet=False):
            assert settings.quiet

        assert settings.quiet

    assert not settings.quiet


def test_quiet_mode_with_exception():
    assert not settings.quiet

    try:
        with quiet_mode():
            assert settings.quiet
            raise Exception("Test exception")
    except Exception:
        pass

    assert not settings.quiet


def test_status_quiet_mode():
    with status("Test message"):
        assert settings.quiet
        # Simulate some work
        pass
    assert not settings.quiet

    with status("Test message"):
        assert settings.quiet
        # Simulate some work
        with quiet_mode():
            assert settings.quiet
            pass
        assert settings.quiet
        pass
    assert not settings.quiet
