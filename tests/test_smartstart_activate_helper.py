from __future__ import annotations

from datetime import datetime, timedelta

from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service
from homeassistant.util import dt as dt_util

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.const import DOMAIN


def test_smartstart_activate_if_needed_calls_switch(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    # Prepare service spy
    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    service_data = {"entity_id": "switch.test"}
    # Call helper: charger_is_on False should cause activation
    res = hass.loop.run_until_complete(
        coord._smartstart_activate_if_needed("Test", "switch.test", service_data, False)
    )
    assert res is True
    assert len(turn_on_calls) == 1

    # If charger already on, nothing should be called
    turn_on_calls.clear()
    res2 = hass.loop.run_until_complete(
        coord._smartstart_activate_if_needed("Test", "switch.test", service_data, True)
    )
    assert res2 is None
    assert len(turn_on_calls) == 0
