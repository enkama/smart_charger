import asyncio
from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util

from custom_components.smart_charger.const import (
    CONF_PRECHARGE_COOLDOWN_MINUTES,
    CONF_PRECHARGE_MIN_DROP_PERCENT,
)
from custom_components.smart_charger.coordinator import (
    DeviceConfig,
    SmartChargerCoordinator,
)


class DummyEntry:
    def __init__(self):
        self.data = {}
        self.options = {}
        self.entry_id = "test"


@pytest.fixture
def hass_loop(event_loop):
    return event_loop


@pytest.mark.asyncio
async def test_precharge_antiflap_min_drop_and_cooldown(monkeypatch, hass):
    # Use the pytest-provided `hass` fixture. Ensure minimal data structures
    # used by the coordinator are present.
    entry = DummyEntry()
    # Ensure hass.data and our domain entries mapping exist
    hass.data.setdefault("smart_charger", {})
    hass.data["smart_charger"].setdefault("entries", {})
    # Ensure config_entries has async_update_entry callable (some test harnesses provide it)
    if not hasattr(hass, "config_entries"):
        hass.config_entries = type(
            "C", (), {"async_update_entry": lambda *a, **k: None}
        )()

    # Inject options: min drop 10% and cooldown 1 minute for test
    entry.options[CONF_PRECHARGE_MIN_DROP_PERCENT] = 10.0
    entry.options[CONF_PRECHARGE_COOLDOWN_MINUTES] = 1.0

    coord = SmartChargerCoordinator(hass, entry)

    # Prepare device config
    device_raw = {
        "name": "car1",
        "battery_sensor": "sensor.car1_batt",
        "charger_switch": "switch.car1",
        "target_level": 90.0,
        "min_level": 20.0,
        "precharge_level": 50.0,
    }
    device = DeviceConfig.from_dict(device_raw)

    # Set initial battery state (release happened at 55%) using HA async state helper
    res = hass.states.async_set(device.battery_sensor, "55")
    if asyncio.iscoroutine(res):
        await res

    # Simulate a precharge release: record last_release_level=55 and set last_release_ts to now
    coord._precharge_last_release_level[device.name] = 55.0
    now = dt_util.utcnow()
    coord._precharge_last_release_ts[device.name] = float(dt_util.as_timestamp(now))

    # Ensure coordinator option attributes are present (normally set via _configure_update_options)
    coord._precharge_min_drop_percent = 10.0
    coord._precharge_cooldown_effective_seconds = 60.0  # 1 minute

    # Case 1: current battery hasn't dropped -> activation should be blocked
    # Ensure coordinator _state snapshot reflects current battery
    coord._state = {device.name: {"battery": 55.0}}

    # Patch _maybe_switch to capture calls; in blocked case it must not be invoked
    called = {}

    async def fake_maybe_switch_block(
        action, service_data, desired=True, bypass_throttle=False, **kw
    ):
        called["action"] = action
        called["desired"] = desired
        return True

    monkeypatch.setattr(coord, "_maybe_switch", fake_maybe_switch_block)

    handled = await coord._apply_charger_precharge_activate_if_needed(
        device,
        charger_ent=device.charger_switch,
        device_name=device.name,
        target_release=55.0,
        forecast_holdoff=False,
        service_data={"entity_id": device.charger_switch},
    )

    # No activation should be attempted and helper should indicate it blocked activation
    assert (
        handled is False
    ), "Activation should be blocked immediately when battery hasn't dropped"
    assert not called, "_maybe_switch should not be called when blocked by min-drop"

    # Case 2: battery drops enough (below 45 => 55 - 10%) -> activation allowed
    # Make sure the cooldown window has expired by moving last_release_ts to the past
    coord._precharge_last_release_ts[device.name] = float(
        dt_util.as_timestamp(dt_util.utcnow() - timedelta(minutes=2))
    )
    coord._state = {device.name: {"battery": 44.0}}

    # Monkeypatch _maybe_switch to capture call instead of performing it
    called = {}

    async def fake_maybe_switch(
        action, service_data, desired=True, bypass_throttle=False, **kw
    ):
        called["action"] = action
        called["desired"] = desired
        return True

    monkeypatch.setattr(coord, "_maybe_switch", fake_maybe_switch)

    handled = await coord._apply_charger_precharge_activate_if_needed(
        device,
        charger_ent=device.charger_switch,
        device_name=device.name,
        target_release=55.0,
        forecast_holdoff=False,
        service_data={"entity_id": device.charger_switch},
    )

    assert (
        handled is True and called.get("action") == "turn_on"
    ), "Activation should be allowed after battery drop and cooldown expired"

    # Case 3: after release but before cooldown expired, activation is blocked even if battery dropped
    # Reset last_release_ts to now and set battery below threshold
    coord._precharge_last_release_ts[device.name] = float(
        dt_util.as_timestamp(dt_util.utcnow())
    )
    coord._state = {device.name: {"battery": 44.0}}
    called.clear()

    handled = await coord._apply_charger_precharge_activate_if_needed(
        device,
        charger_ent=device.charger_switch,
        device_name=device.name,
        target_release=55.0,
        forecast_holdoff=False,
        service_data={"entity_id": device.charger_switch},
    )

    assert handled is False, "Activation should be blocked while cooldown window active"

    # Fast-forward time beyond cooldown (simulate by adjusting last_release_ts to past)
    old_ts = float(dt_util.as_timestamp(dt_util.utcnow() - timedelta(minutes=2)))
    coord._precharge_last_release_ts[device.name] = old_ts

    handled = await coord._apply_charger_precharge_activate_if_needed(
        device,
        charger_ent=device.charger_switch,
        device_name=device.name,
        target_release=55.0,
        forecast_holdoff=False,
        service_data={"entity_id": device.charger_switch},
    )

    assert handled is True, "Activation should be allowed after cooldown expired"
