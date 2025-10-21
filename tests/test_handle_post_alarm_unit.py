"""Direct unit tests for _handle_post_alarm_self_heal helper."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_handle_post_alarm_marks_handled_and_records_correction(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    # Build a pd that represents an alarm in the immediate past and battery below target
    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=1)
    pd = {
        "charger_switch": "switch.unit_test",
        "alarm_time": dt_util.as_local(alarm).isoformat(),
        "target": 90,
        "battery": 10,
        "predicted_level_at_alarm": 50,
    }

    results = {"DeviceX": pd}

    # Call the helper directly
    coordinator._handle_post_alarm_self_heal(results, now)

    # The entry should be marked as handled and a correction should be recorded
    assert "switch.unit_test" in coordinator._post_alarm_last_handled
    assert any(c.get("entity") == "switch.unit_test" for c in coordinator._post_alarm_corrections)
