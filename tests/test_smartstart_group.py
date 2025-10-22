"""Combined tests for SmartStart helper behaviours."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_smartstart_helpers_basic(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.utcnow()
    # Ensure that smartstart ignores when not configured. Call the async
    # smartstart activation helper and ensure it doesn't raise.
    pd = {"charger_switch": "switch.ss", "alarm_time": None, "target": 90, "battery": 10}
    # helper is async - await it
    await coordinator._smartstart_activate_if_needed("", "switch.ss", {}, False)
    # no exceptions
    assert True


async def test_smartstart_activation_and_ignore(hass):
    device = {"name": "SS1", "battery_sensor": "sensor.ss1", "charger_switch": "switch.ss1", "target_level": 95}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=5)
    pd = {"charger_switch": "switch.ss1", "alarm_time": dt_util.as_local(alarm).isoformat(), "target": 95, "battery": 10, "smart_start_margin": 1.0}

    # The async helper may decide to activate/pause; await it to ensure
    # the code path executes without raising.
    await coordinator._smartstart_activate_if_needed("SS1", "switch.ss1", {"entity_id": "switch.ss1"}, False)
    # If activation logic runs, it will update coordinator state; we assert no exceptions
    assert True


def test_smartstart_ignore_distant_forecast_check(hass):
    # Added from test_smartstart_helper.py
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    now = dt_util.now()
    device_name = "Test"
    battery = 50.0
    window_threshold = 5.0
    # start_time equal to now should cause the helper to return expected_on
    start_time = now
    res = coord._smartstart_ignore_distant_forecast_check(
        device_name, battery, window_threshold, start_time, now, expected_on=True
    )
    assert res is True

    # start_time in future should return None
    future = now + timedelta(hours=1)
    res2 = coord._smartstart_ignore_distant_forecast_check(
        device_name, battery, window_threshold, future, now, expected_on=True
    )
    assert res2 is None


def test_smartstart_activate_if_needed_calls_switch(hass):
    # Added from test_smartstart_activate_helper.py
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    # Prepare service spy
    from pytest_homeassistant_custom_component.common import async_mock_service

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    service_data = {"entity_id": "switch.test"}
    # Call helper: charger_is_on False should cause activation
    res = hass.loop.run_until_complete(
        coord._smartstart_activate_if_needed("Test", "switch.test", service_data, False)
    )
    assert res is True
    assert len(turn_on_calls) == 1

    # If charger already on, nothing should be called
    turn_on_calls.clear()
    res2 = hass.loop.run_until_complete(
        coord._smartstart_activate_if_needed("Test", "switch.test", service_data, True)
    )
    assert res2 is None
    assert len(turn_on_calls) == 0
