"""Tests for entity-level adaptive override service."""

from __future__ import annotations

import asyncio

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN


@pytest.mark.asyncio
async def test_set_adaptive_override_entity(hass: HomeAssistant, hass_ws_client):
    """Call the entity-level override service and assert in-memory and persisted mapping."""
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    from custom_components.smart_charger import _register_services
    from custom_components.smart_charger.coordinator import SmartChargerCoordinator

    _register_services(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data[DOMAIN]["entries"][entry.entry_id] = {"entry": entry, "coordinator": coordinator}
    device = entry.data.get("devices", [])
    if device:
        device = device[0]
        entity_id = device.get("charging_sensor") or device.get("presence_sensor") or device.get("name")
    else:
        entity_id = "switch.test_phone_charger"

    # Ensure no per-entity overrides initially
    assert not getattr(coordinator, "_adaptive_throttle_overrides", {})

    # Call entity-level service
    await hass.services.async_call(
        DOMAIN,
        "set_adaptive_override_entity",
        {"entry_id": entry.entry_id, "entity_id": entity_id, "mode": "conservative"},
        blocking=True,
    )
    await asyncio.sleep(0)

    overrides = getattr(coordinator, "_adaptive_throttle_overrides", {}) or {}
    assert entity_id in overrides
    assert overrides[entity_id]["mode"] == "conservative"

    # Check persisted mapping in entry.options
    mapping = entry.options.get("adaptive_mode_overrides") or {}
    assert mapping.get(entity_id) == "conservative"
