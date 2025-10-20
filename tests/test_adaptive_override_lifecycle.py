"""Tests for adaptive override lifecycle: set -> clear and diagnostics exposure."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.diagnostics import _capture_coordinator_state

pytestmark = pytest.mark.asyncio


async def test_override_cleared_and_in_diagnostics(hass) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    # simulate override set
    coordinator._adaptive_mode_override = "aggressive"
    # ensure diagnostics includes it; _capture_coordinator_state only builds insights
    # when coordinator.profiles is truthy, so populate a minimal profile
    coordinator._state = {
        "S1": {
            "battery": 50,
            "target": 95,
            "avg_speed": 10,
            "charging_state": "idle",
            "presence_state": "home",
            "duration_min": 0,
            "start_time": None,
            "alarm_time": None,
            "precharge_level": None,
            "precharge_margin_on": None,
            "precharge_margin_off": None,
            "smart_start_margin": None,
            "precharge_active": False,
            "smart_start_active": False,
            "predicted_level_at_alarm": None,
            "predicted_drain": None,
            "last_update": None,
        }
    }
    _state, _plans, insights = _capture_coordinator_state(coordinator)
    assert insights.get("adaptive_mode_override") == "aggressive"

    # simulate EWMA drop
    coordinator._flipflop_ewma = 0.0
    coordinator._flipflop_ewma_exceeded = False
    coordinator._flipflop_ewma_exceeded_since = None

    # run refresh which should clear override logic in the update path
    await coordinator.async_refresh()

    # override should be cleared
    assert coordinator._adaptive_mode_override is None

    # diagnostics no longer reports the override
    _state, _plans, insights2 = _capture_coordinator_state(coordinator)
    assert insights2.get("adaptive_mode_override") is None
