"""Integration tests for switch confirmation and throttle behavior."""

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


async def _build_plan_for_test(
    coordinator: SmartChargerCoordinator,
    device_config: DeviceConfig,
    when,
    learning,
):
    """Invoke coordinator plan builder with the correct learning window."""

    window = device_config.learning_recent_sample_hours
    if window is None:
        window = getattr(
            coordinator,
            "_default_learning_recent_sample_hours",
            4.0,
        )
    window = max(0.25, min(48.0, float(window)))
    return await coordinator._build_plan(device_config, when, learning, window)


class _LearningStub:
    def avg_speed(self, _profile_id: str | None = None) -> float:
        return 1.0


async def test_confirmation_required_before_switch(hass) -> None:
    """Confirmation count requires consecutive coordinator evaluations before switching."""

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

    # Prepare entity states
    hass.states.async_set("switch.confirm_charger", "off")
    # alarm in the future
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.confirm_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    # First evaluation: desired on, but only first confirmation -> no call
    hass.states.async_set("sensor.confirm_battery", "45")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    # Second evaluation: still desired on -> should call once
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=30))
    hass.states.async_set("sensor.confirm_battery", "44")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # Confirm one turn_on call was made
    assert len(turn_on_calls) == 1
    assert turn_on_calls[0].data["entity_id"] == "switch.confirm_charger"


async def test_throttle_prevents_rapid_toggling(hass) -> None:
    """Throttle should prevent repeated switch calls within the configured window."""

    device_dict = {
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

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    # Prepare states and mocks
    hass.states.async_set("switch.throttle_charger", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.throttle_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")
    turn_off_calls = async_mock_service(hass, "switch", "turn_off")

    # device configuration parsed for debugging; not used directly in this test

    # First: desired on -> should call immediately
    hass.states.async_set("sensor.throttle_battery", "45")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 1
    # Simulate the switch actually turning on (the mocked service doesn't change state)
    hass.states.async_set("switch.throttle_charger", "on")

    # Immediately invert the battery so desired becomes off -> confirmation_count=1 so it would switch
    hass.states.async_set("sensor.throttle_battery", "95")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # Throttle window was 5s, so the turn_off should be suppressed
    assert len(turn_off_calls) == 0

    # After throttle window elapses, second off evaluation should trigger
    # Backdate last switch time so throttle is considered expired (deterministic)
    ent = device_dict[CONF_CHARGER_SWITCH]
    # store as epoch seconds to match coordinator internal expectations
    coordinator._last_switch_time[ent] = dt_util.as_timestamp(dt_util.utcnow() - timedelta(seconds=6))
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=6))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_off_calls) == 1


async def test_force_bypasses_throttle_and_confirmation(hass) -> None:
    """Emergency conditions using force should bypass throttle/confirmation."""

    device_dict = {
        "name": "ForceVehicle",
        CONF_BATTERY_SENSOR: "sensor.force_battery",
        CONF_CHARGER_SWITCH: "switch.force_charger",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.force_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_THROTTLE_SECONDS: 60,
        CONF_SWITCH_CONFIRMATION_COUNT: 5,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.force_charger", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.force_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    # Simulate emergency: battery below min_level -> coordinator should call turn_on with force=True
    hass.states.async_set("sensor.force_battery", "10")
    await coordinator.async_refresh()
    await hass.async_block_till_done()

    assert len(turn_on_calls) == 1


async def test_confirmation_counter_resets_on_opposite_observation(hass) -> None:
    """If a different desired state is observed the confirmation counter resets."""

    device_dict = {
        "name": "ResetVehicle",
        CONF_BATTERY_SENSOR: "sensor.reset_battery",
        CONF_CHARGER_SWITCH: "switch.reset_charger",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.reset_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 3,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("switch.reset_charger", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.reset_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    # First two observations indicate desired on (but less than 3 required)
    hass.states.async_set("sensor.reset_battery", "45")
    # Ensure per-device confirmation count is registered in the coordinator (deterministic)
    ent = device_dict[CONF_CHARGER_SWITCH]
    coordinator._device_switch_throttle[f"{ent}::confirm"] = float(
        device_dict.get(
            CONF_SWITCH_CONFIRMATION_COUNT, coordinator._confirmation_required
        )
    )
    coordinator._desired_state_history.setdefault(ent, (False, 0))
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # After first observation we should have counted 1
    assert coordinator._desired_state_history.get(ent) == (True, 1)

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
    hass.states.async_set("sensor.reset_battery", "44")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # After second observation we should have counted 2
    assert coordinator._desired_state_history.get(ent) == (True, 2)

    # Now an opposite observation; counter should reset
    hass.states.async_set("sensor.reset_battery", "95")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # After opposite observation the counter should have reset to 1 for False
    assert coordinator._desired_state_history.get(ent) == (False, 1)

    # Another on observation must restart counting; no turn_on yet
    hass.states.async_set("sensor.reset_battery", "43")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # After restarting counting we expect no switch call
    assert coordinator._desired_state_history.get(ent) == (True, 1)
    assert len(turn_on_calls) == 0


async def test_multi_device_independence(hass) -> None:
    """Multiple devices should have independent throttle and confirmation state."""

    device_a = {
        "name": "A",
        CONF_BATTERY_SENSOR: "sensor.a_batt",
        CONF_CHARGER_SWITCH: "switch.a_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.a_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 2,
    }

    device_b = {
        "name": "B",
        CONF_BATTERY_SENSOR: "sensor.b_batt",
        CONF_CHARGER_SWITCH: "switch.b_chg",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.b_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
        CONF_SWITCH_CONFIRMATION_COUNT: 2,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_a, device_b]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)
    hass.states.async_set("switch.a_chg", "off")
    hass.states.async_set("switch.b_chg", "off")
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.a_alarm", future_alarm.isoformat())
    hass.states.async_set("sensor.b_alarm", future_alarm.isoformat())

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    # parsed devices not needed for assertions here

    # First evaluation: both want to turn on -> no calls yet
    hass.states.async_set("sensor.a_batt", "45")
    hass.states.async_set("sensor.b_batt", "45")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(turn_on_calls) == 0

    # Second evaluation: both still want to turn on -> both should result in calls
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
    hass.states.async_set("sensor.a_batt", "44")
    hass.states.async_set("sensor.b_batt", "44")
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    # Two calls should be present, one per device
    assert len(turn_on_calls) == 2
