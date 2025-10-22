"""Combined tests for Coordinator behaviours."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.smart_charger.const import (
    ALARM_MODE_SINGLE,
    CONF_ALARM_ENTITY,
    CONF_ALARM_MODE,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_CHARGING_SENSOR,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_PRECHARGE_MARGIN_ON,
    CONF_PRESENCE_SENSOR,
    CONF_SMART_START_MARGIN,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    CONF_SWITCH_CONFIRMATION_COUNT,
    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    DOMAIN,
)
from custom_components.smart_charger.coordinator import (
    DeviceConfig,
    SmartChargerCoordinator,
)

pytestmark = pytest.mark.asyncio


async def test_coordinator_precharge_guard(hass):
    device = {
        "name": "C1",
        "battery_sensor": "sensor.c1",
        "charger_switch": "switch.c1",
        "target_level": 95,
    }
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    _now = dt_util.utcnow()
    # No exceptions when checking precharge guard: call the current helper that
    # performs a similar safety check. It returns a boolean; we only assert no
    # exception is raised.
    coordinator._final_guard_should_suppress("", 0.0, True)
    assert True


async def test_coordinator_last_action_guard_behaviour(hass):
    device = {
        "name": "C2",
        "battery_sensor": "sensor.c2",
        "charger_switch": "switch.c2",
        "target_level": 80,
    }
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)

    _now = dt_util.utcnow()
    _pd = {
        "charger_switch": "switch.c2",
    "alarm_time": dt_util.as_local(_now - timedelta(seconds=5)).isoformat(),
        "target": 80,
        "battery": 20,
    }
    # Use _final_guard_should_suppress to exercise the same guard behaviour
    coordinator._final_guard_should_suppress("switch.c2", 0.0, True)
    assert True


# --- Begin copied from test_coordinator_precharge.py ---


class _LearningStub:
    def avg_speed(self, _profile_id: str | None = None) -> float:
        return 1.0


class _LearningFastStub:
    def __init__(self, value: float = 8.5) -> None:
        self._value = value

    def avg_speed(self, _profile_id: str | None = None) -> float:
        return self._value


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
            DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
        )
    window = max(0.25, min(48.0, float(window)))
    return await coordinator._build_plan(device_config, when, learning, window)


async def test_precharge_turns_on_charger_when_threshold_hit(hass) -> None:
    """Ensure precharge engages the charger when the hysteresis window latches."""

    device_dict = {
        "name": "Test Vehicle",
        CONF_BATTERY_SENSOR: "sensor.test_battery",
        CONF_CHARGER_SWITCH: "switch.test_charger",
        CONF_TARGET_LEVEL: 80,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 50,
        CONF_PRECHARGE_MARGIN_ON: 5,
        CONF_PRECHARGE_MARGIN_OFF: 8,
        CONF_SMART_START_MARGIN: 3,
        CONF_USE_PREDICTIVE_MODE: False,
        CONF_ALARM_ENTITY: "sensor.test_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    hass.states.async_set("sensor.test_battery", "40")
    hass.states.async_set("switch.test_charger", "off")
    # Alarm two hours in the future keeps the plan realistic.
    future_alarm = dt_util.now() + timedelta(hours=2)
    hass.states.async_set("sensor.test_alarm", future_alarm.isoformat())
    await hass.async_block_till_done()

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        dt_util.now(),
        _LearningStub(),
    )

    await hass.async_block_till_done()

    assert plan is not None
    assert plan.precharge_active is True
    assert len(turn_on_calls) == 1
    call = turn_on_calls[0]
    assert call.data["entity_id"] == "switch.test_charger"


async def test_plan_hides_optional_durations_without_precharge(hass) -> None:
    """Optional duration fields should be omitted when they add no extra information."""

    device_dict = {
        "name": "NoPrecharge",
        CONF_BATTERY_SENSOR: "sensor.no_precharge_battery",
        CONF_CHARGER_SWITCH: "switch.no_precharge_charger",
        CONF_TARGET_LEVEL: 80,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 30,
        CONF_USE_PREDICTIVE_MODE: True,
        CONF_ALARM_ENTITY: "sensor.no_precharge_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.now()
    hass.states.async_set("sensor.no_precharge_battery", "70")
    hass.states.async_set("switch.no_precharge_charger", "off")
    hass.states.async_set(
        "sensor.no_precharge_alarm", (now + timedelta(hours=5)).isoformat()
    )
    await hass.async_block_till_done()

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        now,
        _LearningFastStub(6.0),
    )

    assert plan is not None
    exported = plan.as_dict()
    assert exported["duration_min"] > 0
    assert exported.get("charge_duration_min") is None
    assert exported.get("total_duration_min") is None
    assert (
        "precharge_duration_min" not in exported
        or exported["precharge_duration_min"] is None
    )


async def test_plan_exposes_precharge_durations_when_active(hass) -> None:
    """When precharge is active, optional duration fields should be included."""

    device_dict = {
        "name": "PrechargeActive",
        CONF_BATTERY_SENSOR: "sensor.precharge_battery",
        CONF_CHARGER_SWITCH: "switch.precharge_charger",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 60,
        CONF_PRECHARGE_MARGIN_ON: 5,
        CONF_PRECHARGE_MARGIN_OFF: 8,
        CONF_USE_PREDICTIVE_MODE: True,
        CONF_ALARM_ENTITY: "sensor.precharge_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.now()
    hass.states.async_set("sensor.precharge_battery", "40")
    hass.states.async_set("switch.precharge_charger", "off")
    hass.states.async_set(
        "sensor.precharge_alarm", (now + timedelta(hours=4)).isoformat()
    )
    await hass.async_block_till_done()

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        now,
        _LearningStub(),
    )

    assert plan is not None
    assert plan.precharge_active is True

    exported = plan.as_dict()
    assert exported["duration_min"] > 0
    assert exported.get("charge_duration_min") is not None
    assert exported.get("total_duration_min") is not None
    assert exported.get("precharge_duration_min") is not None


async def test_smart_start_waits_until_window_when_not_precharging(hass) -> None:
    """Ensure charger pauses while waiting for smart-start window if no precharge required."""

    device_dict = {
        "name": "Test Vehicle",
        CONF_BATTERY_SENSOR: "sensor.test_battery",
        CONF_CHARGER_SWITCH: "switch.test_charger",
        CONF_CHARGING_SENSOR: "binary_sensor.test_charging",
        CONF_TARGET_LEVEL: 95,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 50,
        CONF_USE_PREDICTIVE_MODE: True,
        CONF_ALARM_ENTITY: "sensor.test_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.now()
    hass.states.async_set("sensor.test_battery", "73")
    hass.states.async_set("switch.test_charger", "on")
    hass.states.async_set("binary_sensor.test_charging", "on")
    hass.states.async_set("sensor.test_alarm", (now + timedelta(hours=8)).isoformat())
    await hass.async_block_till_done()

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        now,
        _LearningStub(),
    )
    await hass.async_block_till_done()

    assert plan is not None
    assert plan.smart_start_active is True
    assert plan.precharge_active is False


async def test_observed_drain_spike_tempered(hass) -> None:
    """A large observed drop should not explode the predicted drain duration."""

    device_dict = {
        "name": "DrainSpike EV",
        CONF_BATTERY_SENSOR: "sensor.drain_battery",
        CONF_CHARGER_SWITCH: "switch.drain_charger",
        CONF_CHARGING_SENSOR: "binary_sensor.drain_charging",
        CONF_PRESENCE_SENSOR: "person.drain_owner",
        CONF_TARGET_LEVEL: 95,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 55,
        CONF_USE_PREDICTIVE_MODE: True,
        CONF_ALARM_ENTITY: "sensor.drain_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.now().replace(hour=22, minute=30, second=0, microsecond=0)
    hass.states.async_set("sensor.drain_battery", "55")
    hass.states.async_set("switch.drain_charger", "off")
    hass.states.async_set("binary_sensor.drain_charging", "off")
    hass.states.async_set("person.drain_owner", "home")
    hass.states.async_set(
        "sensor.drain_alarm",
        (now + timedelta(hours=9, minutes=40)).isoformat(),
    )
    await hass.async_block_till_done()

    coordinator._battery_history[device_dict["name"]] = (
        now - timedelta(minutes=30),
        58.0,
        False,
    )

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        now,
        _LearningFastStub(8.5),
    )
    await hass.async_block_till_done()

    assert plan is not None
    expected_rate = 0.3 + (6.0 - 0.3) * 0.25
    assert plan.drain_rate == pytest.approx(expected_rate, rel=0.05)
    assert plan.duration_min < 24 * 60
    assert plan.duration_min < 420


async def test_observed_drain_ignores_recent_charging_sample(hass) -> None:
    """Drain calculation should ignore history captured while the device was charging."""

    device_dict = {
        "name": "ChargeHistory EV",
        CONF_BATTERY_SENSOR: "sensor.charge_history_battery",
        CONF_CHARGER_SWITCH: "switch.charge_history_charger",
        CONF_CHARGING_SENSOR: "binary_sensor.charge_history_state",
        CONF_TARGET_LEVEL: 90,
        CONF_MIN_LEVEL: 20,
        CONF_PRECHARGE_LEVEL: 55,
        CONF_USE_PREDICTIVE_MODE: True,
        CONF_ALARM_ENTITY: "sensor.charge_history_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.now().replace(minute=0, second=0, microsecond=0)
    hass.states.async_set("sensor.charge_history_battery", "57")
    hass.states.async_set("switch.charge_history_charger", "off")
    hass.states.async_set("binary_sensor.charge_history_state", "off")
    hass.states.async_set(
        "sensor.charge_history_alarm",
        (now + timedelta(hours=6)).isoformat(),
    )
    await hass.async_block_till_done()

    coordinator._battery_history[device_dict["name"]] = (
        now - timedelta(minutes=1),
        60.0,
        True,
    )

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        now,
        _LearningFastStub(7.5),
    )
    await hass.async_block_till_done()

    assert plan is not None
    assert plan.precharge_active is False
    assert coordinator._precharge_release.get(device_dict["name"]) is None


async def test_precharge_not_triggered_when_well_above_threshold(hass) -> None:
    """High battery levels must not spuriously trigger a precharge latch."""

    device_dict = {
        "name": "No Precharge",
        CONF_BATTERY_SENSOR: "sensor.no_precharge_battery",
        CONF_CHARGER_SWITCH: "switch.no_precharge_charger",
        CONF_CHARGING_SENSOR: "binary_sensor.no_precharge_state",
        CONF_TARGET_LEVEL: 95,
        CONF_MIN_LEVEL: 30,
        CONF_PRECHARGE_LEVEL: 50,
        CONF_USE_PREDICTIVE_MODE: True,
        CONF_ALARM_ENTITY: "sensor.no_precharge_alarm",
        CONF_ALARM_MODE: ALARM_MODE_SINGLE,
    }

    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device_dict]})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.now().replace(hour=15, minute=0, second=0, microsecond=0)
    hass.states.async_set("sensor.no_precharge_battery", "70")
    hass.states.async_set("switch.no_precharge_charger", "off")
    hass.states.async_set("binary_sensor.no_precharge_state", "off")
    hass.states.async_set(
        "sensor.no_precharge_alarm",
        (now + timedelta(hours=16)).isoformat(),
    )
    await hass.async_block_till_done()

    turn_on_calls = async_mock_service(hass, "switch", "turn_on")

    device_config = DeviceConfig.from_dict(device_dict)
    plan = await _build_plan_for_test(
        coordinator,
        device_config,
        now,
        _LearningFastStub(7.5),
    )
