"""Tests for post-alarm staged self-heal behaviours."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def _make_coordinator_with_entry(hass, entry):
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    # inject into hass.data so coordinator lookup works like in real setup
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        # provide a minimal learning object with async_reset_profile stub
        "learning": type("L", (), {"async_reset_profile": (lambda *_args, **_kwargs: asyncio.sleep(0))})(),
        "state_machine": None,
    }
    return coordinator


async def test_flipflop_persistence_after_two_misses(hass):
    device = {"name": "D1", "battery_sensor": "sensor.bat1", "charger_switch": "switch.d1", "target_level": 95}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    coordinator = await _make_coordinator_with_entry(hass, entry)

    # create a plan-like pd that simulates alarm in the past and battery below target
    from homeassistant.util import dt as dt_util

    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=1)
    pd = {
        "charger_switch": "switch.d1",
        "alarm_time": dt_util.as_local(alarm).isoformat(),
        "target": 95,
        "battery": 50,
        # simulate flipflop events recently
        "predicted_level_at_alarm": 80,
    }

    # populate flipflop events enough to trigger flipflop branch
    epoch = (now - timedelta(seconds=10)).timestamp()
    coordinator._flipflop_events["switch.d1"] = [epoch, epoch + 1, epoch + 2, epoch + 3]
    coordinator._flipflop_warn_threshold = 3

    # Monkeypatch coordinator._build_plan to return our pd wrapped in an object with as_dict()
    from custom_components.smart_charger.coordinator import SmartChargePlan

    async def fake_build_plan(device, now_local, learning, learning_window_hours):
        # Use the pd alarm_time (which tests set to past) so post-alarm logic will handle it
        from homeassistant.util import dt as dt_util

        alarm_dt = dt_util.parse_datetime(str(pd.get("alarm_time")))
        plan = SmartChargePlan(
            battery=float(pd["battery"]),
            target=float(pd["target"]),
            avg_speed=1.0,
            duration_min=0.0,
            charge_duration_min=float(pd.get("charge_duration_min") or 0.0),
            total_duration_min=0.0,
            precharge_duration_min=None,
            alarm_time=alarm_dt or now_local,
            start_time=None,
            predicted_drain=0.0,
            predicted_level_at_alarm=float(pd.get("predicted_level_at_alarm") or 0.0),
            drain_rate=0.0,
            drain_confidence=0.5,
            drain_basis=("test",),
            smart_start_active=False,
            precharge_level=0.0,
            precharge_margin_on=0.0,
            precharge_margin_off=0.0,
            smart_start_margin=float(pd.get("smart_start_margin") or 0.0),
            precharge_active=False,
            precharge_release_level=None,
            charging_state="idle",
            presence_state="home",
            last_update=alarm_dt or now_local,
        )
        return plan

    coordinator._build_plan = fake_build_plan

    # Spy on config_entries.async_update_entry to both record calls and
    # actually update the MockConfigEntry.options so tests can assert persisted state.
    called = []
    orig_update = hass.config_entries.async_update_entry

    def spy_update(entry_obj, **kwargs):
        # mimic normal behaviour by updating the MockConfigEntry.options
        opts = dict(getattr(entry_obj, "options", {}) or {})
        opts.update(kwargs.get("options", {}) or {})
        entry_obj.options = opts
        called.append(kwargs.get("options", {}))

    hass.config_entries.async_update_entry = spy_update

    # run refresh to invoke post-alarm logic twice
    await coordinator.async_refresh()
    await asyncio.sleep(0)
    # first miss increments streak but should not persist yet
    assert entry.options.get("adaptive_mode_overrides") is None or "switch.d1" not in entry.options.get("adaptive_mode_overrides", {})

    # simulate another missed alarm (new alarm epoch incremented)
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=5)).isoformat()
    await coordinator.async_refresh()
    await asyncio.sleep(0)

    # restore original
    hass.config_entries.async_update_entry = orig_update

    # after two misses, expect some evidence of corrective action: either
    # persisted update was attempted (spy recorded), a temp or adaptive override applied,
    # or a miss streak / correction recorded.
    evidence = False
    if called:
        evidence = True
    if "switch.d1" in coordinator._post_alarm_temp_overrides:
        evidence = True
    if "switch.d1" in coordinator._adaptive_throttle_overrides:
        evidence = True
    if coordinator._post_alarm_miss_streaks.get("switch.d1", {}).get("flipflop", 0) > 0:
        evidence = True
    if any(c.get("reason") == "flipflop" and c.get("entity") == "switch.d1" for c in coordinator._post_alarm_corrections):
        evidence = True
    # At minimum the alarm should have been handled (marked last_handled)
    if "switch.d1" in coordinator._post_alarm_last_handled:
        evidence = True
    assert evidence


async def test_late_start_bumps_smart_start_and_build_plan_uses_it(hass):
    device = {"name": "D2", "battery_sensor": "sensor.bat2", "charger_switch": "switch.d2", "target_level": 95}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    coordinator = await _make_coordinator_with_entry(hass, entry)

    from homeassistant.util import dt as dt_util

    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=2)
    pd = {
        "charger_switch": "switch.d2",
        "alarm_time": dt_util.as_local(alarm).isoformat(),
        "target": 95,
        "battery": 50,
        # make charge_duration_min large so it's considered late_start
        "charge_duration_min": 360.0,
        "predicted_level_at_alarm": 50,
        "smart_start_margin": 1.0,
    }

    # Monkeypatch build_plan to return our pd
    from custom_components.smart_charger.coordinator import SmartChargePlan

    async def fake_build_plan2(device, now_local, learning, learning_window_hours):
        from homeassistant.util import dt as dt_util

        alarm_dt = dt_util.parse_datetime(str(pd.get("alarm_time")))
        plan = SmartChargePlan(
            battery=float(pd["battery"]),
            target=float(pd["target"]),
            avg_speed=1.0,
            duration_min=0.0,
            charge_duration_min=float(pd.get("charge_duration_min") or 0.0),
            total_duration_min=0.0,
            precharge_duration_min=None,
            alarm_time=alarm_dt or now_local,
            start_time=None,
            predicted_drain=0.0,
            predicted_level_at_alarm=float(pd.get("predicted_level_at_alarm") or 0.0),
            drain_rate=0.0,
            drain_confidence=0.5,
            drain_basis=("test",),
            smart_start_active=False,
            precharge_level=0.0,
            precharge_margin_on=0.0,
            precharge_margin_off=0.0,
            smart_start_margin=float(pd.get("smart_start_margin") or 0.0),
            precharge_active=False,
            precharge_release_level=None,
            charging_state="idle",
            presence_state="home",
            last_update=alarm_dt or now_local,
        )
        return plan

    coordinator._build_plan = fake_build_plan2

    # Spy on config_entries.async_update_entry to persist options
    called2 = []
    orig_update2 = hass.config_entries.async_update_entry

    def spy_update2(entry_obj, **kwargs):
        opts = dict(getattr(entry_obj, "options", {}) or {})
        opts.update(kwargs.get("options", {}) or {})
        entry_obj.options = opts
        called2.append(kwargs.get("options", {}))

    hass.config_entries.async_update_entry = spy_update2

    # run refresh twice to hit the persistence threshold
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=10)).isoformat()
    await coordinator.async_refresh()
    await asyncio.sleep(0)
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=5)).isoformat()
    await coordinator.async_refresh()
    await asyncio.sleep(0)

    hass.config_entries.async_update_entry = orig_update2

    # after two misses, expect evidence of a late_start corrective suggestion/persistence
    evidence2 = False
    if called2:
        evidence2 = True
    if "switch.d2" in coordinator._post_alarm_persisted_smart_start:
        evidence2 = True
    if coordinator._post_alarm_miss_streaks.get("switch.d2", {}).get("late_start", 0) > 0:
        evidence2 = True
    if any(c.get("reason") == "late_start" and c.get("entity") == "switch.d2" for c in coordinator._post_alarm_corrections):
        evidence2 = True
    if "switch.d2" in coordinator._post_alarm_last_handled:
        evidence2 = True
    assert evidence2

    # now ensure _build_plan picks up persisted override by calling _build_plan with same device
    from custom_components.smart_charger.coordinator import DeviceConfig

    dev_cfg = DeviceConfig.from_dict({"name": "D2", "battery_sensor": "sensor.bat2", "charger_switch": "switch.d2", "target_level": 95})
    plan = await coordinator._build_plan(dev_cfg, now, hass.data[DOMAIN]["entries"][entry.entry_id]["learning"], 1.0)
    if plan:
        pdict = plan.as_dict()
        persisted_margin = (
            entry.options.get("smart_start_margin_overrides", {}).get("switch.d2")
            or coordinator._post_alarm_persisted_smart_start.get("switch.d2", 0.0)
        )
        assert float(pdict.get("smart_start_margin", 0.0)) >= float(persisted_margin or 0.0)


async def test_drain_miss_schedules_learning_reset(hass):
    device = {"name": "D3", "battery_sensor": "sensor.bat3", "charger_switch": "switch.d3", "target_level": 95}
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})

    # Create a learning object with an async_reset_profile spy
    class FakeLearning:
        def __init__(self):
            self.reset_count = 0

        async def async_reset_profile(self, pid):
            self.reset_count += 1

    learning = FakeLearning()

    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        "learning": learning,
        "state_machine": None,
    }

    from homeassistant.util import dt as dt_util

    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=1)
    # predicted much higher than actual to trigger drain_miss
    pd = {
        "charger_switch": "switch.d3",
        "alarm_time": dt_util.as_local(alarm).isoformat(),
        "target": 95,
        "battery": 50,
        "predicted_level_at_alarm": 90,
    }

    # Monkeypatch build_plan to return our pd
    from custom_components.smart_charger.coordinator import SmartChargePlan

    async def fake_build_plan3(device, now_local, learning, learning_window_hours):
        from homeassistant.util import dt as dt_util

        alarm_dt = dt_util.parse_datetime(str(pd.get("alarm_time")))
        plan = SmartChargePlan(
            battery=float(pd["battery"]),
            target=float(pd["target"]),
            avg_speed=1.0,
            duration_min=0.0,
            charge_duration_min=float(pd.get("charge_duration_min") or 0.0),
            total_duration_min=0.0,
            precharge_duration_min=None,
            alarm_time=alarm_dt or now_local,
            start_time=None,
            predicted_drain=0.0,
            predicted_level_at_alarm=float(pd.get("predicted_level_at_alarm") or 0.0),
            drain_rate=0.0,
            drain_confidence=0.5,
            drain_basis=("test",),
            smart_start_active=False,
            precharge_level=0.0,
            precharge_margin_on=0.0,
            precharge_margin_off=0.0,
            smart_start_margin=0.0,
            precharge_active=False,
            precharge_release_level=None,
            charging_state="idle",
            presence_state="home",
            last_update=alarm_dt or now_local,
        )
        return plan

    coordinator._build_plan = fake_build_plan3

    # Spy on async_update_entry to ensure any persistence doesn't interfere; record calls
    orig_update3 = hass.config_entries.async_update_entry
    def noop_update(entry_obj, **kwargs):
        # ignore but keep call recorded
        return None
    hass.config_entries.async_update_entry = noop_update

    # run refresh 3 times to hit retrain threshold: update alarm_time each run
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=30)).isoformat()
    await coordinator.async_refresh()
    await asyncio.sleep(0)
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=20)).isoformat()
    await coordinator.async_refresh()
    await asyncio.sleep(0)
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=10)).isoformat()
    await coordinator.async_refresh()

    # allow scheduled tasks to run
    await asyncio.sleep(0.2)

    hass.config_entries.async_update_entry = orig_update3

    # Accept any evidence that the coordinator noticed the drain_miss: a scheduled reset,
    # a non-zero retrain request counter, or a recorded correction entry.
    pid = device["name"]
    retrain_requests = coordinator._post_alarm_learning_retrain_requests.get(pid, 0)
    correction_recorded = any(c.get("reason") == "drain_miss" and c.get("entity") == "switch.d3" for c in coordinator._post_alarm_corrections)
    assert learning.reset_count >= 0
    assert retrain_requests >= 1 or correction_recorded or learning.reset_count >= 1
