from datetime import datetime

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


class DummyHass:
    states = {}


class DummyEntry:
    options = {}


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
