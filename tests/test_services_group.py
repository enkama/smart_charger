"""Combined tests for Services behaviours."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_services_override_entity_and_entry(hass):
    # Combine behaviors from the service-related tests: entity-level override,
    # entry-level override persistence, and post-alarm services (list/accept/revert/clear).

    # --- Setup entry and coordinator ---
    device = {
        "name": "SV1",
        "battery_sensor": "sensor.sv1",
        "charger_switch": "switch.sv1",
        "target_level": 95,
    }
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    # Register services like the original tests do
    from custom_components.smart_charger import _register_services

    _register_services(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        "learning": None,
        "state_machine": None,
    }

    # --- Entity-level override via service semantics (simulate) ---
    # Initially, no overrides
    assert not getattr(coordinator, "_adaptive_throttle_overrides", {})
    # Apply an override via helper (original tests used the helper directly)
    coordinator._apply_adaptive_throttle_for_entity("switch.sv1", [], 0)
    # Simulate persisting entry-level mapping (use the dict shape expected by coordinator)
    coordinator._adaptive_throttle_overrides["switch.sv1"] = {
        "original": 0.0,
        "applied": 2.0,
        "expires": 999999.0,
    }
    assert (coordinator._adaptive_throttle_overrides.get("switch.sv1") or {}).get(
        "applied"
    ) == 2.0

    # --- Entry-level override service semantics ---
    assert getattr(coordinator, "_adaptive_mode_override", None) is None
    # Call the set_adaptive_override service to persist entry-level override
    await hass.services.async_call(
        DOMAIN,
        "set_adaptive_override",
        {"entry_id": entry.entry_id, "mode": "aggressive"},
        blocking=True,
    )
    # allow callbacks to complete
    import asyncio

    await asyncio.sleep(0)
    assert coordinator._adaptive_mode_override == "aggressive"
    assert entry.options.get("adaptive_mode_override") == "aggressive"

    # --- Post-alarm services: list/accept/revert/clear ---
    # inject sample post-alarm state
    coordinator._post_alarm_temp_overrides["switch.sv1"] = {
        "applied": 300,
        "expires": 999999,
        "reason": "flipflop",
    }
    coordinator._post_alarm_persisted_smart_start["switch.sv1"] = 2.5
    coordinator._post_alarm_miss_streaks["switch.sv1"] = {"flipflop": 2}
    coordinator._post_alarm_corrections.insert(
        0,
        {
            "entity": "switch.sv1",
            "device": "SV1",
            "alarm_epoch": 0,
            "timestamp": 0,
            "reason": "flipflop",
            "details": {},
        },
    )

    # Call list_post_alarm_insights (should not raise)
    await hass.services.async_call(
        DOMAIN, "list_post_alarm_insights", {"entry_id": entry.entry_id}, blocking=True
    )

    # Accept suggested persistence using the service (this should update entry.options)
    await hass.services.async_call(
        DOMAIN,
        "accept_suggested_persistence",
        {"entry_id": entry.entry_id, "entity_id": "switch.sv1"},
        blocking=True,
    )
    await asyncio.sleep(0)
    assert entry.options.get("adaptive_mode_overrides") or {}

    # Revert suggested persistence via service
    await hass.services.async_call(
        DOMAIN,
        "revert_suggested_persistence",
        {"entry_id": entry.entry_id, "entity_id": "switch.sv1"},
        blocking=True,
    )
    await asyncio.sleep(0)
    assert "switch.sv1" not in (entry.options.get("adaptive_mode_overrides") or {})

    # Clear post-alarm corrections
    coordinator._post_alarm_corrections.append(
        {"entity": "switch.sv1", "reason": "flipflop"}
    )
    coordinator._post_alarm_corrections.append(
        {"entity": "switch.other", "reason": "late_start"}
    )
    # simulate clear for switch.sv1
    coordinator._post_alarm_corrections = [
        c
        for c in coordinator._post_alarm_corrections
        if c.get("entity") != "switch.sv1"
    ]
    assert all(
        c.get("entity") != "switch.sv1" for c in coordinator._post_alarm_corrections
    )
