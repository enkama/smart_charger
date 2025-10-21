"""Tests for diagnostics exposing post-alarm telemetry and corrections."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN

pytestmark = pytest.mark.asyncio


async def test_diagnostics_includes_post_alarm(hass):
    from custom_components.smart_charger.coordinator import SmartChargerCoordinator
    from custom_components.smart_charger.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "devices": [
                {
                    "name": "D1",
                    "battery_sensor": "sensor.b1",
                    "charger_switch": "switch.d1",
                }
            ]
        },
        options={},
    )
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        "learning": None,
        "state_machine": None,
    }

    # add telemetry and corrections
    coordinator._flipflop_events["switch.d1"] = [1, 2, 3]
    coordinator._post_alarm_corrections.append(
        {"entity": "switch.d1", "reason": "flipflop"}
    )
    # Provide a minimal internal state so diagnostics will include coordinator_insights
    coordinator._state = {
        "D1": {
            "alarm_time": "2025-10-21T00:00:00+00:00",
            "battery": 50,
            "target": 95,
            "smart_start_active": False,
            "precharge_active": False,
            "predicted_level_at_alarm": 80,
            "predicted_drain": 0.0,
            "last_update": "2025-10-21T00:00:00+00:00",
            "charger_switch": "switch.d1",
        }
    }

    diag = await async_get_config_entry_diagnostics(hass, entry)
    insights = diag.get("coordinator_insights", {})
    assert "post_alarm_corrections" in insights
    assert insights.get("flipflop_events", {}).get("switch.d1") == 3
