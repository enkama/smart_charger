from datetime import datetime

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


class DummyConfigEntries:
    def __init__(self):
        self.last_update = None

    def async_update_entry(self, entry, options=None):
        # record the options provided for inspection
        self.last_update = dict(options or {})


class DummyHass:
    def __init__(self):
        self.config_entries = DummyConfigEntries()
        self.states = {}


class DummyEntry:
    def __init__(self):
        self.options = {}


def test_update_flipflop_ewma_and_mode():
    hass = DummyHass()
    entry = DummyEntry()
    coord = SmartChargerCoordinator(hass, entry)

    now = datetime.utcnow()
    now_epoch = float(now.timestamp())

    # Seed many recent events to produce a high rate
    coord._flipflop_window_seconds = 10.0
    # several events within window
    coord._flipflop_events = {"e1": [now_epoch - 1, now_epoch - 2, now_epoch - 3], "e2": [now_epoch - 1, now_epoch - 2]}

    # Ensure default alpha is used
    coord._flipflop_ewma = 0.0

    # Call helper
    coord._update_flipflop_ewma_and_mode(now_epoch)

    # After update, EWMA should be set and exceeded flag may be True depending on thresholds
    assert isinstance(coord._flipflop_ewma, float)
    # If we set the override, ensure config_entries was updated or internal state set
    if coord._adaptive_mode_override == "aggressive":
        assert hass.config_entries.last_update is not None
    else:
        # no aggressive override yet, ensure state tracked
        assert coord._flipflop_ewma_last_update is not None
