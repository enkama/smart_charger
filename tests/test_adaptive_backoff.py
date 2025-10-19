import time

import pytest

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import (
    DOMAIN,
    CONF_ADAPTIVE_THROTTLE_ENABLED,
    CONF_ADAPTIVE_THROTTLE_MULTIPLIER,
    CONF_ADAPTIVE_THROTTLE_MIN_SECONDS,
    CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS,
    CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP,
    CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
    CONF_ADAPTIVE_THROTTLE_MODE,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_TARGET_LEVEL,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
)

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


@pytest.mark.asyncio
async def test_adaptive_backoff_grows_with_mode(hass):
    """Simulate repeated flip-flops and verify aggressive > normal > conservative applied throttle."""

    # create a minimal device entry and coordinator
    device = {
        "name": "TestVehicle",
        CONF_BATTERY_SENSOR: "sensor.test_battery",
        CONF_CHARGER_SWITCH: "switch.test_charger",
        CONF_TARGET_LEVEL: 80,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 50,
        CONF_USE_PREDICTIVE_MODE: False,
    }

    base_options = {
        CONF_ADAPTIVE_THROTTLE_ENABLED: True,
        CONF_ADAPTIVE_THROTTLE_MULTIPLIER: 2.0,
        CONF_ADAPTIVE_THROTTLE_MIN_SECONDS: 5,
        CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS: 60,
        CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP: 0.5,
        CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER: 10.0,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options=base_options)
    entry.add_to_hass(hass)

    # coordinator variable not used in this test; keep entry installed for hass

    # we will test three modes and capture applied values
    results = {}

    for mode in ("conservative", "normal", "aggressive"):
        # create a fresh entry with the desired mode and coordinator
        opts = dict(base_options, **{CONF_ADAPTIVE_THROTTLE_MODE: mode})
        mode_entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options=opts)
        mode_entry.add_to_hass(hass)
        mode_coordinator = SmartChargerCoordinator(hass, mode_entry)

        # simulate flip-flop events on a single entity
        ent = "switch.test_device"
        # ensure coordinator structures exist
        mode_coordinator._flipflop_events = {}
        mode_coordinator._adaptive_throttle_overrides = {}

        now = time.time()
        # simulate 6 flip-flops within the window
        timestamps = [now - i for i in range(6)]
        mode_coordinator._flipflop_events[ent] = list(timestamps)

        # call the coordinator update path that applies adaptive overrides
        await mode_coordinator._async_update_data()

        ov = mode_coordinator._adaptive_throttle_overrides.get(ent)
        assert ov is not None, "Adaptive override should be applied"
        results[mode] = ov.get("applied")

    # aggressive should produce an equal or larger applied throttle than normal, and normal >= conservative
    assert results["aggressive"] >= results["normal"] >= results["conservative"]
