import time
from datetime import datetime, timedelta

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


class DummyHass:
    states = {}


class DummyEntry:
    options = {}


def test_prune_flipflop_events():
    hass = DummyHass()
    entry = DummyEntry()
    coord = SmartChargerCoordinator(hass, entry)

    now = datetime.utcnow()
    now_epoch = float((now - timedelta(seconds=0)).timestamp())

    # Seed events: one old, two recent
    old = now_epoch - 1000.0
    recent1 = now_epoch - 10.0
    recent2 = now_epoch - 2.0

    coord._flipflop_window_seconds = 60.0
    coord._flipflop_events = {
        "entity.test": [old, recent1, recent2],
        "entity.empty": [],
    }

    coord._prune_flipflop_events(now_epoch)

    assert "entity.test" in coord._flipflop_events
    assert coord._flipflop_events["entity.test"] == [recent1, recent2]
    # empty list should be removed
    assert "entity.empty" not in coord._flipflop_events
