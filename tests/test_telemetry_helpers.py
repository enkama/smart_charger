from datetime import datetime, timedelta

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


class DummyHass:
    states = {}


class DummyEntry:
    options = {}


class DummyConfigEntries:
    def __init__(self):
        self.last_update = None

    def async_update_entry(self, entry, options=None):
        # record the options provided for inspection
        self.last_update = dict(options or {})


class DummyHassWithConfig:
    def __init__(self):
        self.config_entries = DummyConfigEntries()
        self.states = {}


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


def test_apply_adaptive_throttle_creates_override():
    hass = DummyHass()
    entry = DummyEntry()
    coord = SmartChargerCoordinator(hass, entry)

    now = datetime.utcnow()
    now_epoch = float(now.timestamp())

    # Configure coordinator thresholds to make behavior deterministic
    coord._flipflop_warn_threshold = 2
    coord._adaptive_throttle_multiplier = 2.0
    coord._adaptive_throttle_backoff_step = 0.5
    coord._adaptive_throttle_max_multiplier = 10.0
    coord._adaptive_throttle_min_seconds = 10.0
    coord._adaptive_throttle_duration_seconds = 300.0

    # Seed current throttle and events
    entity = "test.entity"
    coord._device_switch_throttle[entity] = 5.0
    recent = [now_epoch - 1.0, now_epoch - 2.0, now_epoch - 3.0]

    # Apply
    coord._apply_adaptive_throttle_for_entity(entity, recent, now_epoch)

    assert entity in coord._adaptive_throttle_overrides
    meta = coord._adaptive_throttle_overrides[entity]
    assert meta["applied"] >= coord._adaptive_throttle_min_seconds
    assert coord._device_switch_throttle[entity] == meta["applied"]


def test_update_flipflop_ewma_and_mode():
    hass = DummyHassWithConfig()
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
