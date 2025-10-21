from __future__ import annotations

import pytest
from datetime import datetime
from types import SimpleNamespace

from custom_components.smart_charger.coordinator import (
    DeviceConfig,
    SmartChargerCoordinator,
)
from custom_components.smart_charger.const import DOMAIN, DEFAULT_SWITCH_CONFIRMATION_COUNT

pytestmark = pytest.mark.asyncio


class _DummyPlan:
    def __init__(self, payload: dict):
        self._payload = payload

    def as_dict(self) -> dict:
        return dict(self._payload)


class _LearningStub:
    def __init__(self):
        self.last_window = None

    def set_recent_sample_window(self, w):
        self.last_window = w


async def test_prepare_device_and_build_plan_sets_throttle_and_returns_plan(hass):
    device_dict = {
        "name": "PrepareTest",
        "battery_sensor": "sensor.test_batt",
        "charger_switch": "switch.test_charger",
        "target_level": 80,
        "min_level": 20,
        "precharge_level": 50,
        "use_predictive_mode": True,
        "switch_throttle_seconds": 12,
        "switch_confirmation_count": 3,
    }

    entry = SimpleNamespace(entry_id="testid", data={"devices": [device_dict]}, options={})
    # create coordinator
    coordinator = SmartChargerCoordinator(hass, entry)

    # stub _build_plan to return a DummyPlan
    async def _fake_build_plan(device, now_local, learning, window):
        return _DummyPlan({"battery": 42, "window": window})

    coordinator._build_plan = _fake_build_plan

    learning = _LearningStub()
    device = DeviceConfig.from_dict(device_dict)

    pd = await coordinator._prepare_device_and_build_plan(device, datetime.utcnow(), learning)

    assert pd is not None
    assert pd.get("battery") == 42
    assert pd.get("charger_switch") == device.charger_switch
    # learning window set
    assert learning.last_window is not None
    # per-entity throttle configured
    assert coordinator._device_switch_throttle.get(device.charger_switch) == pytest.approx(12.0)
    # confirmation stored (uses confirm key)
    assert coordinator._device_switch_throttle.get(f"{device.charger_switch}::confirm") == pytest.approx(3.0)


async def test_prepare_device_and_build_plan_uses_defaults_when_missing(hass):
    device_dict = {
        "name": "PrepareDefaults",
        "battery_sensor": "sensor.test_batt2",
        "charger_switch": "switch.test_charger2",
        "target_level": 80,
        "min_level": 20,
        "precharge_level": 50,
        "use_predictive_mode": True,
        # omit switch_throttle_seconds and switch_confirmation_count
    }

    entry = SimpleNamespace(entry_id="testid2", data={"devices": [device_dict]}, options={})
    coordinator = SmartChargerCoordinator(hass, entry)

    async def _fake_build_plan(device, now_local, learning, window):
        return _DummyPlan({"battery": 30, "window": window})

    coordinator._build_plan = _fake_build_plan
    device = DeviceConfig.from_dict(device_dict)

    pd = await coordinator._prepare_device_and_build_plan(device, datetime.utcnow(), None)
    assert pd is not None
    # default throttle should be set on the ent
    assert coordinator._device_switch_throttle.get(device.charger_switch) == pytest.approx(coordinator._default_switch_throttle_seconds)
    # default confirmation should be present
    assert coordinator._device_switch_throttle.get(f"{device.charger_switch}::confirm") == pytest.approx(coordinator._confirmation_required)


async def test_prepare_device_and_build_plan_learning_errors_are_swallowed(hass):
    device_dict = {
        "name": "PrepareLearnErr",
        "battery_sensor": "sensor.test_batt3",
        "charger_switch": "switch.test_charger3",
        "target_level": 80,
        "min_level": 20,
        "precharge_level": 50,
        "use_predictive_mode": True,
    }

    entry = SimpleNamespace(entry_id="testid3", data={"devices": [device_dict]}, options={})
    coordinator = SmartChargerCoordinator(hass, entry)

    async def _fake_build_plan(device, now_local, learning, window):
        return _DummyPlan({"battery": 10})

    coordinator._build_plan = _fake_build_plan

    class BadLearning:
        def set_recent_sample_window(self, w):
            raise RuntimeError("boom")

    device = DeviceConfig.from_dict(device_dict)
    # Should not raise despite learning.set_recent_sample_window raising
    pd = await coordinator._prepare_device_and_build_plan(device, datetime.utcnow(), BadLearning())
    assert pd is not None

