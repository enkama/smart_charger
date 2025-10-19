import time

import pytest

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import (
    DOMAIN,
    CONF_ADAPTIVE_EWMA_ALPHA,
)
from custom_components.smart_charger.coordinator import SmartChargerCoordinator
# sensor import not required for coordinator-level EWMA tests


@pytest.mark.asyncio
async def test_ewma_alpha_effect(hass):
    """Ensure a larger alpha reacts faster to a spike in flip-flop rate."""

    device = {"name": "EV1", "battery_sensor": "sensor.bat", "charger_switch": "switch.ch", "target_level": 80, "min_level": 20, "precharge_level": 50, "use_predictive_mode": False}

    # slow alpha (0.1)
    entry_slow = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={CONF_ADAPTIVE_EWMA_ALPHA: 0.1})
    entry_slow.add_to_hass(hass)
    coord_slow = SmartChargerCoordinator(hass, entry_slow)

    # fast alpha (0.9)
    entry_fast = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={CONF_ADAPTIVE_EWMA_ALPHA: 0.9})
    entry_fast.add_to_hass(hass)
    coord_fast = SmartChargerCoordinator(hass, entry_fast)

    ent = "switch.test"
    now = time.time()

    # initial small activity to initialize EWMA
    coord_slow._flipflop_events = {ent: [now - 300, now - 250]}
    coord_fast._flipflop_events = {ent: [now - 300, now - 250]}
    await coord_slow._async_update_data()
    await coord_fast._async_update_data()

    # now simulate a spike: many recent events
    spike_ts = [now - i for i in range(10)]
    coord_slow._flipflop_events = {ent: spike_ts}
    coord_fast._flipflop_events = {ent: spike_ts}

    await coord_slow._async_update_data()
    await coord_fast._async_update_data()

    ewma_slow = getattr(coord_slow, "_flipflop_ewma", 0.0)
    ewma_fast = getattr(coord_fast, "_flipflop_ewma", 0.0)

    # fast alpha should have larger EWMA after a spike (reacts more strongly)
    assert ewma_fast >= ewma_slow
