"""Combined tests for adaptive behaviour: backoff, lifecycle, and aggressive EWMA."""

from __future__ import annotations

import time
from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import (
    CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP,
    CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS,
    CONF_ADAPTIVE_THROTTLE_ENABLED,
    CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
    CONF_ADAPTIVE_THROTTLE_MIN_SECONDS,
    CONF_ADAPTIVE_THROTTLE_MODE,
    CONF_ADAPTIVE_THROTTLE_MULTIPLIER,
    DOMAIN,
)
from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.diagnostics import _capture_coordinator_state

pytestmark = pytest.mark.asyncio


async def test_adaptive_backoff_grows_with_mode(hass):
    """Simulate repeated flip-flops and verify aggressive > normal > conservative applied throttle."""

    # create a minimal device entry and coordinator
    device = {
        "name": "TestVehicle",
        "battery_sensor": "sensor.test_battery",
        "charger_switch": "switch.test_charger",
        "target_level": 80,
        "min_level": 20,
        "precharge_level": 50,
        "use_predictive_mode": False,
    }

    base_options = {
        CONF_ADAPTIVE_THROTTLE_ENABLED: True,
        CONF_ADAPTIVE_THROTTLE_MULTIPLIER: 2.0,
        CONF_ADAPTIVE_THROTTLE_MIN_SECONDS: 5,
        CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS: 60,
        CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP: 0.5,
        CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER: 10.0,
    }

    entry = MockConfigEntry(
        domain=DOMAIN, data={"devices": [device]}, options=base_options
    )
    entry.add_to_hass(hass)

    results = {}

    for mode in ("conservative", "normal", "aggressive"):
        opts = dict(base_options, **{CONF_ADAPTIVE_THROTTLE_MODE: mode})
        mode_entry = MockConfigEntry(
            domain=DOMAIN, data={"devices": [device]}, options=opts
        )
        mode_entry.add_to_hass(hass)
        mode_coordinator = SmartChargerCoordinator(hass, mode_entry)

        ent = "switch.test_device"
        mode_coordinator._flipflop_events = {}
        mode_coordinator._adaptive_throttle_overrides = {}

        now = time.time()
        timestamps = [now - i for i in range(6)]
        mode_coordinator._flipflop_events[ent] = list(timestamps)

        await mode_coordinator._async_update_data()

        ov = mode_coordinator._adaptive_throttle_overrides.get(ent)
        assert ov is not None, "Adaptive override should be applied"
        results[mode] = ov.get("applied")

    assert results["aggressive"] >= results["normal"] >= results["conservative"]


async def test_override_cleared_and_in_diagnostics(hass) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    coordinator._adaptive_mode_override = "aggressive"
    coordinator._state = {
        "S1": {
            "battery": 50,
            "target": 95,
            "avg_speed": 10,
            "charging_state": "idle",
            "presence_state": "home",
            "duration_min": 0,
            "start_time": None,
            "alarm_time": None,
            "precharge_level": None,
            "precharge_margin_on": None,
            "precharge_margin_off": None,
            "smart_start_margin": None,
            "precharge_active": False,
            "smart_start_active": False,
            "predicted_level_at_alarm": None,
            "predicted_drain": None,
            "last_update": None,
        }
    }
    _state, _plans, insights = _capture_coordinator_state(coordinator)
    assert insights.get("adaptive_mode_override") == "aggressive"

    coordinator._flipflop_ewma = 0.0
    coordinator._flipflop_ewma_exceeded = False
    coordinator._flipflop_ewma_exceeded_since = None

    await coordinator.async_refresh()

    assert coordinator._adaptive_mode_override is None

    _state, _plans, insights2 = _capture_coordinator_state(coordinator)
    assert insights2.get("adaptive_mode_override") is None


async def test_sustained_ewma_triggers_aggressive(hass) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    assert coordinator._adaptive_mode_override is None

    ent = "switch.testcharger"
    now = dt_util.as_timestamp(dt_util.utcnow())
    warn = getattr(coordinator, "_flipflop_warn_threshold", 3)

    timestamps = [now - 1.0 * i for i in range(warn + 2)]
    coordinator._flipflop_events[ent] = timestamps

    await coordinator.async_refresh()

    coordinator._flipflop_ewma_exceeded_since = dt_util.as_timestamp(
        dt_util.utcnow() - timedelta(seconds=301)
    )
    coordinator._flipflop_ewma = 1.0
    coordinator._flipflop_ewma_exceeded = True

    await coordinator.async_refresh()

    assert coordinator._adaptive_mode_override == "aggressive"
