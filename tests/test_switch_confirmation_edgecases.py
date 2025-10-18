"""Edge-case tests for switch confirmation and throttle behavior.

These tests validate rapid alternation (should not reach confirmation)
and that a newly created coordinator still enforces the configured
confirmation count (i.e. confirmation requirement is part of config
and is re-registered on init).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
    async_fire_time_changed,
)

from custom_components.smart_charger.const import (
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_TARGET_LEVEL,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    CONF_ALARM_ENTITY,
    CONF_ALARM_MODE,
    ALARM_MODE_SINGLE,
    CONF_SWITCH_CONFIRMATION_COUNT,
    CONF_SWITCH_THROTTLE_SECONDS,
    DOMAIN,
)
from custom_components.smart_charger.coordinator import DeviceConfig, SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_rapid_alternation_prevents_confirmation(hass) -> None:
    """Rapidly alternating desired state should never reach the consecutive confirmation count."""

    device_dict = {
        "name": "Alternator",
        CONF_BATTERY_SENSOR: "sensor.alt_batt",
        CONF_CHARGER_SWITCH: "switch.alt_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.alt_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 2,
        # keep throttle low so it does not interfere with confirmation-focused test
        CONF_SWITCH_THROTTLE_SECONDS: 1,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.alt_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.alt_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    # Alternate desired state several times; confirmation requires 2 consecutive
    for _ in range(4):
        # desired ON
        hass.states.async_set("sensor.alt_batt", "45")
        await coordinator.async_refresh()
        await hass.async_block_till_done()
        # Immediately flip to desired OFF
        hass.states.async_set("sensor.alt_batt", "95")
        await coordinator.async_refresh()
        await hass.async_block_till_done()

    # No turn_on calls should have occurred because confirmations never reached 2
    assert len(turn_on_calls) == 0


async def test_confirmation_enforced_after_coordinator_reinit(hass) -> None:
    """A newly created coordinator must still enforce the configured confirmation count.

    This ensures the per-device confirmation config is re-registered on init.
    """

    device_dict = {
        "name": "Persist",
        CONF_BATTERY_SENSOR: "sensor.persist_batt",
        CONF_CHARGER_SWITCH: "switch.persist_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.persist_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 2,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    # Initial coordinator: perform one observation (should not switch)
    coord1 = SmartChargerCoordinator(hass, entry)
    hass.states.async_set("switch.persist_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.persist_alarm", future_alarm.isoformat())
    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    hass.states.async_set("sensor.persist_batt", "45")
    await coord1.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    # Recreate coordinator (simulating a reload/restart). The confirmation
    # configuration should be applied again; one observation should still
    # not be sufficient to call the switch.
    coord2 = SmartChargerCoordinator(hass, entry)
    hass.states.async_set("sensor.persist_batt", "45")
    await coord2.async_refresh()
    await hass.async_block_till_done()
    # Still no call after a single observation on the new coordinator
    assert len(turn_on_calls) == 0

    # Second consecutive observation should trigger the switch
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
    hass.states.async_set("sensor.persist_batt", "44")
    await coord2.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1
