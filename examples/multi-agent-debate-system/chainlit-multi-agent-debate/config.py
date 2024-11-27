import langroid.utils.configuration

def get_global_settings(debug=False, nocache=False):
    """
    Returns global settings for Langroid.

    Args:
        debug (bool): Enable or disable debug mode.
        nocache (bool): Enable or disable caching.

    Returns:
        Settings: Langroid configuration settings.
    """
    return langroid.utils.configuration.Settings(
        debug=debug,
        cache=not nocache,
    )

