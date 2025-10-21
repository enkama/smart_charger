"""Tests for entry-level adaptive override services."""

from __future__ import annotations

import asyncio

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN


@pytest.mark.asyncio
async def test_set_and_clear_adaptive_override(hass: HomeAssistant, hass_ws_client):
    """Call services to set and clear the adaptive override and assert persistence."""
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    # Register services (without running full async_setup_entry which triggers HA checks)
    from custom_components.smart_charger import _register_services
    from custom_components.smart_charger.coordinator import SmartChargerCoordinator

    _register_services(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data[DOMAIN]["entries"][entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
    }
    assert getattr(coordinator, "_adaptive_mode_override", None) is None

    # Call set_adaptive_override service
    await hass.services.async_call(
        DOMAIN,
        "set_adaptive_override",
        {"entry_id": entry.entry_id, "mode": "aggressive"},
        blocking=True,
    )
    await asyncio.sleep(0)

    assert coordinator._adaptive_mode_override == "aggressive"
    assert entry.options.get("adaptive_mode_override") == "aggressive"

    # Call clear_adaptive_override service
    await hass.services.async_call(
        DOMAIN,
        "clear_adaptive_override",
        {"entry_id": entry.entry_id},
        blocking=True,
    )
    await asyncio.sleep(0)

    assert coordinator._adaptive_mode_override is None
    assert entry.options.get("adaptive_mode_override") is None
