"""Combined tests for Coordinator behaviours."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_coordinator_precharge_guard(hass):
    device = {"name": "C1", "battery_sensor": "sensor.c1", "charger_switch": "switch.c1", "target_level": 95}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.utcnow()
    # No exceptions when checking precharge guard
    coordinator._check_last_action_guard({}, now)
    assert True


async def test_coordinator_last_action_guard_behaviour(hass):
    device = {"name": "C2", "battery_sensor": "sensor.c2", "charger_switch": "switch.c2", "target_level": 80}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.utcnow()
    pd = {"charger_switch": "switch.c2", "alarm_time": dt_util.as_local(now - timedelta(seconds=5)).isoformat(), "target": 80, "battery": 20}
    coordinator._check_last_action_guard({"C2": pd}, now)
    assert True
