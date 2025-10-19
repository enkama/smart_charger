"""Regression test: coordinator must not overwrite _last_action_state before final throttle check."""
from __future__ import annotations

from __future__ import annotations

"""Regression test: coordinator must not overwrite _last_action_state before final throttle check."""

from datetime import timedelta

from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
from homeassistant.util import dt as dt_util

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.const import (
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_MIN_LEVEL,
    CONF_TARGET_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_ALARM_ENTITY,
    CONF_ALARM_MODE,
    CONF_SWITCH_THROTTLE_SECONDS,
    CONF_SWITCH_CONFIRMATION_COUNT,
    CONF_USE_PREDICTIVE_MODE,
    ALARM_MODE_SINGLE,
    DOMAIN,
)


async def test_last_action_state_preserved_until_throttle_check(hass) -> None:
    """Ensure _last_action_state is not overwritten before final throttle check."""

    device = {
        "name": "GuardVehicle",
        CONF_BATTERY_SENSOR: "sensor.guard_batt",
        CONF_CHARGER_SWITCH: "switch.guard_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_ALARM_ENTITY: "sensor.guard_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_THROTTLE_SECONDS: 10,
        CONF_SWITCH_CONFIRMATION_COUNT: 1,
        CONF_USE_PREDICTIVE_MODE: False,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]})
    entry.add_to_hass(hass)

    coord = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.guard_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.guard_alarm", future_alarm.isoformat())

    turn_on = async_mock_service(hass, "switch", "turn_on")
    turn_off = async_mock_service(hass, "switch", "turn_off")

    # First request turns it on
    hass.states.async_set("sensor.guard_batt", "45")
    await coord.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on) == 1

    # Simulate the switch was turned on externally
    hass.states.async_set("switch.guard_chg", "on")

    # Immediately invert desired -> coordinator would desire off, but throttle should prevent immediate call
    hass.states.async_set("sensor.guard_batt", "95")

    # Before refresh, record the current stored last_action_state
    prev_last_action = coord._last_action_state.get("switch.guard_chg")

    await coord.async_refresh()
    await hass.async_block_till_done()

    # Immediately after evaluation, the previous last_action_state must have been used for throttle decision
    # The coordinator should not have overwritten the previous_last_action prior to the final throttle check.
    # final action should be suppressed because throttle window still active
    assert len(turn_off) == 0
    assert prev_last_action is True
