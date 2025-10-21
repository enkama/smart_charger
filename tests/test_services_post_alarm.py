"""Tests for post-alarm related services: list/clear/accept/revert."""

from __future__ import annotations

import asyncio

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN

pytestmark = pytest.mark.asyncio


async def test_list_and_clear_post_alarm_history_and_accept_revert(hass):
    from custom_components.smart_charger import _register_services
    from custom_components.smart_charger.coordinator import SmartChargerCoordinator

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [{"name": "D1", "battery_sensor": "sensor.b1", "charger_switch": "switch.d1"}]}, options={})
    entry.add_to_hass(hass)
    _register_services(hass)

    coordinator = SmartChargerCoordinator(hass, entry)
    # inject into hass.data entries mapping like async_setup_entry would
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        "learning": None,
        "state_machine": None,
    }

    # simulate a staged suggestion: temp override and persisted smart start
    coordinator._post_alarm_temp_overrides["switch.d1"] = {"applied": 300, "expires": 999999, "reason": "flipflop"}
    coordinator._post_alarm_persisted_smart_start["switch.d1"] = 2.5
    coordinator._post_alarm_miss_streaks["switch.d1"] = {"flipflop": 2}
    coordinator._post_alarm_corrections.insert(0, {"entity": "switch.d1", "device": "D1", "alarm_epoch": 0, "timestamp": 0, "reason": "flipflop", "details": {}})

    # Call list_post_alarm_insights (should not raise)
    await hass.services.async_call(DOMAIN, "list_post_alarm_insights", {"entry_id": entry.entry_id}, blocking=True)

    # Accept suggested persistence for entity switch.d1
    await hass.services.async_call(DOMAIN, "accept_suggested_persistence", {"entry_id": entry.entry_id, "entity_id": "switch.d1"}, blocking=True)
    await asyncio.sleep(0)

    # Check entry.options updated
    assert entry.options.get("adaptive_mode_overrides") or entry.options.get("smart_start_margin_overrides")

    # Revert suggested persistence for entity switch.d1
    await hass.services.async_call(DOMAIN, "revert_suggested_persistence", {"entry_id": entry.entry_id, "entity_id": "switch.d1"}, blocking=True)
    await asyncio.sleep(0)

    # Ensure options cleared for that entity
    sm = entry.options.get("smart_start_margin_overrides") or {}
    am = entry.options.get("adaptive_mode_overrides") or {}
    assert "switch.d1" not in sm and "switch.d1" not in am

    # Now test clear_post_alarm_history service
    # add two corrections back
    coordinator._post_alarm_corrections.append({"entity": "switch.d1", "reason": "flipflop"})
    coordinator._post_alarm_corrections.append({"entity": "switch.d2", "reason": "late_start"})
    await hass.services.async_call(DOMAIN, "clear_post_alarm_history", {"entry_id": entry.entry_id, "entity_id": "switch.d1"}, blocking=True)
    # only switch.d2 should remain
    assert all(c.get("entity") != "switch.d1" for c in coordinator._post_alarm_corrections)

    # clear all
    coordinator._post_alarm_corrections.append({"entity": "switch.d3", "reason": "drain_miss"})
    await hass.services.async_call(DOMAIN, "clear_post_alarm_history", {"entry_id": entry.entry_id}, blocking=True)
    assert not coordinator._post_alarm_corrections
