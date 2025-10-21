import time

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import CONF_ADAPTIVE_EWMA_ALPHA, DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator


@pytest.mark.asyncio
async def test_ewma_converges_to_rate(hass):
    """EWMA should approach a sustained rate over repeated updates."""

    device = {
        "name": "EV1",
        "battery_sensor": "sensor.bat",
        "charger_switch": "switch.ch",
        "target_level": 80,
        "min_level": 20,
        "precharge_level": 50,
        "use_predictive_mode": False,
    }

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"devices": [device]},
        options={CONF_ADAPTIVE_EWMA_ALPHA: 0.2},
    )
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    ent = "switch.test"
    now = time.time()

    # Simulate repeated steady activity: 30 events in the window
    spike_ts = [now - i for i in range(30)]

    # Apply several updates to let EWMA settle
    for _ in range(20):
        coord._flipflop_events = {ent: spike_ts}
        await coord._async_update_data()

    ewma = getattr(coord, "_flipflop_ewma", 0.0)

    # compute expected steady rate per second roughly (events/window). Use window from coordinator
    window = getattr(coord, "_flipflop_window_seconds", 300.0)
    expected_rate = len(spike_ts) / float(window)

    # EWMA should be close to the sustained rate (tolerance reasonable due to alpha)
    assert abs(ewma - expected_rate) / max(expected_rate, 1e-6) < 0.5


@pytest.mark.asyncio
async def test_ewma_decays_with_alpha(hass):
    """After a spike, EWMA should decay faster for larger alpha."""

    device = {
        "name": "EV1",
        "battery_sensor": "sensor.bat",
        "charger_switch": "switch.ch",
        "target_level": 80,
        "min_level": 20,
        "precharge_level": 50,
        "use_predictive_mode": False,
    }

    entry_low = MockConfigEntry(
        domain=DOMAIN,
        data={"devices": [device]},
        options={CONF_ADAPTIVE_EWMA_ALPHA: 0.1},
    )
    entry_low.add_to_hass(hass)
    coord_low = SmartChargerCoordinator(hass, entry_low)

    entry_high = MockConfigEntry(
        domain=DOMAIN,
        data={"devices": [device]},
        options={CONF_ADAPTIVE_EWMA_ALPHA: 0.8},
    )
    entry_high.add_to_hass(hass)
    coord_high = SmartChargerCoordinator(hass, entry_high)

    ent = "switch.test"
    now = time.time()

    # Create spike
    spike_ts = [now - i for i in range(20)]
    coord_low._flipflop_events = {ent: spike_ts}
    coord_high._flipflop_events = {ent: spike_ts}
    await coord_low._async_update_data()
    await coord_high._async_update_data()

    # Now set to zero activity and apply several updates
    coord_low._flipflop_events = {ent: []}
    coord_high._flipflop_events = {ent: []}

    # collect EWMA after several decays
    for _ in range(10):
        await coord_low._async_update_data()
        await coord_high._async_update_data()

    ewma_low = getattr(coord_low, "_flipflop_ewma", 0.0)
    ewma_high = getattr(coord_high, "_flipflop_ewma", 0.0)

    # higher alpha should have decayed more toward zero
    assert ewma_high <= ewma_low
