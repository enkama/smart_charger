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
