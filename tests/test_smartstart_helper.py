from __future__ import annotations

from datetime import timedelta

from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant.util import dt as dt_util

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.const import DOMAIN


def test_smartstart_ignore_distant_forecast_check(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    now = dt_util.now()
    device_name = "Test"
    battery = 50.0
    window_threshold = 5.0
    # start_time equal to now should cause the helper to return expected_on
    start_time = now
    res = coord._smartstart_ignore_distant_forecast_check(
        device_name, battery, window_threshold, start_time, now, expected_on=True
    )
    assert res is True

    # start_time in future should return None
    future = now + timedelta(hours=1)
    res2 = coord._smartstart_ignore_distant_forecast_check(
        device_name, battery, window_threshold, future, now, expected_on=True
    )
    assert res2 is None
