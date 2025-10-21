from datetime import datetime, timedelta

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


class DummyHass:
    states = {}


class DummyEntry:
    options = {}


def test_record_flipflop_event_trims_old():
    hass = DummyHass()
    entry = DummyEntry()
    coord = SmartChargerCoordinator(hass, entry)

    now = datetime.utcnow()
    now_epoch = float(now.timestamp())

    ent = "switch.test"
    coord._flipflop_window_seconds = 30.0
    # seed with an old event and ensure trimming keeps only recent
    coord._flipflop_events = {ent: [now_epoch - 100.0]}

    coord._record_flipflop_event(ent, now_epoch)

    assert ent in coord._flipflop_events
    assert all(e >= now_epoch - coord._flipflop_window_seconds for e in coord._flipflop_events[ent])
