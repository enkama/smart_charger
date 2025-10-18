"""Basic defaults tests."""

from custom_components.smart_charger.const import DEFAULT_SWITCH_CONFIRMATION_COUNT


def test_default_confirmation_count_is_one() -> None:
    """Protect against accidental changes to the default confirmation count."""
    assert DEFAULT_SWITCH_CONFIRMATION_COUNT == 1
