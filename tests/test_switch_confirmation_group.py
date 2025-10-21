"""Grouped tests for switch-confirmation and throttle behaviors."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
    async_mock_service,
)

from custom_components.smart_charger.const import (
    ALARM_MODE_SINGLE,
    CONF_ALARM_ENTITY,
    CONF_ALARM_MODE,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_PRECHARGE_MARGIN_ON,
    CONF_SWITCH_CONFIRMATION_COUNT,
    CONF_SWITCH_THROTTLE_SECONDS,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    DOMAIN,
)
from custom_components.smart_charger.coordinator import (
    DeviceConfig,
    SmartChargerCoordinator,
)

pytestmark = pytest.mark.asyncio


async def test_rapid_alternation_prevents_confirmation(hass) -> None:
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
        CONF_SWITCH_THROTTLE_SECONDS: 1,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.alt_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.alt_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    for _ in range(4):
        hass.states.async_set("sensor.alt_batt", "45")
        await coordinator.async_refresh()
        await hass.async_block_till_done()
        hass.states.async_set("sensor.alt_batt", "95")
        await coordinator.async_refresh()
        await hass.async_block_till_done()

    assert len(turn_on_calls) == 0


async def test_confirmation_enforced_after_coordinator_reinit(hass) -> None:
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

    coord1 = SmartChargerCoordinator(hass, entry)
    hass.states.async_set("switch.persist_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.persist_alarm", future_alarm.isoformat())
    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    hass.states.async_set("sensor.persist_batt", "45")
    await coord1.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    coord2 = SmartChargerCoordinator(hass, entry)
    hass.states.async_set("sensor.persist_batt", "45")
    await coord2.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
    hass.states.async_set("sensor.persist_batt", "44")
    await coord2.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1


async def test_confirmation_required_before_switch_and_throttle(hass) -> None:
    device_dict = {
        "name": "ConfirmVehicle",
        CONF_BATTERY_SENSOR: "sensor.confirm_battery",
        CONF_CHARGER_SWITCH: "switch.confirm_charger",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_PRECHARGE_MARGIN_ON: 5,
        CONF_PRECHARGE_MARGIN_OFF: 8,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.confirm_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 2,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.confirm_charger", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.confirm_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    hass.states.async_set("sensor.confirm_battery", "45")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=30))
    hass.states.async_set("sensor.confirm_battery", "44")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1

    # Throttle prevention test
    device_dict2 = {
        "name": "ThrottleVehicle",
        CONF_BATTERY_SENSOR: "sensor.throttle_battery",
        CONF_CHARGER_SWITCH: "switch.throttle_charger",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.throttle_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_THROTTLE_SECONDS: 5,
        CONF_SWITCH_CONFIRMATION_COUNT: 1,
    }
    entry2 = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict2]})
    entry2.add_to_hass(hass)
    coord2 = SmartChargerCoordinator(hass, entry2)
    hass.states.async_set("switch.throttle_charger", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.throttle_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")
    turn_off_calls = async_mock_service(hass, "switch", "turn_off")

    hass.states.async_set("sensor.throttle_battery", "45")
    await coord2.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1

    hass.states.async_set("sensor.throttle_battery", "95")
    await coord2.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_off_calls) == 0

    ent = device_dict2[CONF_CHARGER_SWITCH]
    coord2._last_switch_time[ent] = dt_util.as_timestamp(
        dt_util.utcnow() - timedelta(seconds=6)
    )
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=6))
    await hass.async_block_till_done()
    await coord2.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_off_calls) == 1
