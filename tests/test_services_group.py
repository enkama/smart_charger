"""Combined tests for Services behaviours."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_services_override_entity_and_entry(hass):
    device = {"name": "SV1", "battery_sensor": "sensor.sv1", "charger_switch": "switch.sv1", "target_level": 95}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    # Test overriding an entity
    coordinator._apply_adaptive_throttle_for_entity("switch.sv1", [], 0)
    # Test entry level override storage
    coordinator._adaptive_throttle_overrides["switch.sv1"] = 2.0
    assert coordinator._adaptive_throttle_overrides.get("switch.sv1") == 2.0
