"""Deterministic unit tests for switch confirmation and throttle behavior.

These tests avoid sleeps by driving the coordinator with async_refresh and
time jumps (async_fire_time_changed) and rely on the coordinator's logical
time handling to be deterministic in tests.
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

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.coordinator import DeviceConfig
from custom_components.smart_charger.const import (
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_TARGET_LEVEL,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_ALARM_ENTITY,
    CONF_ALARM_MODE,
    ALARM_MODE_SINGLE,
    CONF_SWITCH_THROTTLE_SECONDS,
    CONF_SWITCH_CONFIRMATION_COUNT,
    DOMAIN,
)

pytestmark = pytest.mark.asyncio


async def test_confirmation_debounce_triggers_on_second_refresh(hass) -> None:
    """Coordinator must require configured consecutive confirmations before switching."""

    device = {
        "name": "UnitConfirm",
        CONF_BATTERY_SENSOR: "sensor.uc_batt",
        CONF_CHARGER_SWITCH: "switch.uc_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_ALARM_ENTITY: "sensor.uc_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 2,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]})
    entry.add_to_hass(hass)

    coord = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.uc_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.uc_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    # First observation desires ON but should not call yet (confirmation=2)
    hass.states.async_set("sensor.uc_batt", "45")
    await coord.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    # Advance time and run second observation: now it should call once
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
    hass.states.async_set("sensor.uc_batt", "44")
    await coord.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1


async def test_throttle_suppresses_quick_second_call(hass) -> None:
    """Coordinator should suppress a second call occurring inside throttle window."""

    device = {
        "name": "UnitThrottle",
        CONF_BATTERY_SENSOR: "sensor.ut_batt",
        CONF_CHARGER_SWITCH: "switch.ut_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_ALARM_ENTITY: "sensor.ut_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_THROTTLE_SECONDS: 5,
        CONF_SWITCH_CONFIRMATION_COUNT: 1,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]})
    entry.add_to_hass(hass)

    coord = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.ut_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.ut_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")
    turn_off_calls = async_mock_service(hass, "switch", "turn_off")

    # First: desired on -> immediate call
    hass.states.async_set("sensor.ut_batt", "45")
    await coord.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1

    # Immediately desire OFF -> would normally call but throttle should suppress
    hass.states.async_set("sensor.ut_batt", "95")
    await coord.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_off_calls) == 0

    # Advance time beyond throttle and try again -> now allowed
    coord._last_switch_time[device[CONF_CHARGER_SWITCH]] = dt_util.utcnow() - timedelta(seconds=6)
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=6))
    hass.states.async_set("sensor.ut_batt", "95")
    await coord.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_off_calls) == 1
