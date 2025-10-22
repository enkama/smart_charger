"""Grouped tests for post-alarm behaviours and diagnostics."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_diagnostics_includes_post_alarm(hass):
    from custom_components.smart_charger.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "devices": [
                {
                    "name": "D1",
                    "battery_sensor": "sensor.b1",
                    "charger_switch": "switch.d1",
                }
            ]
        },
        options={},
    )
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        "learning": None,
        "state_machine": None,
    }

    coordinator._flipflop_events["switch.d1"] = [1, 2, 3]
    coordinator._post_alarm_corrections.append(
        {"entity": "switch.d1", "reason": "flipflop"}
    )
    coordinator._state = {
        "D1": {
            "alarm_time": "2025-10-21T00:00:00+00:00",
            "battery": 50,
            "target": 95,
            "smart_start_active": False,
            "precharge_active": False,
            "predicted_level_at_alarm": 80,
            "predicted_drain": 0.0,
            "last_update": "2025-10-21T00:00:00+00:00",
            "charger_switch": "switch.d1",
        }
    }

    diag = await async_get_config_entry_diagnostics(hass, entry)
    insights = diag.get("coordinator_insights", {})
    assert "post_alarm_corrections" in insights
    assert insights.get("flipflop_events", {}).get("switch.d1") == 3


async def test_handle_post_alarm_marks_handled_and_records_correction(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=1)
    pd = {
        "charger_switch": "switch.unit_test",
        "alarm_time": dt_util.as_local(alarm).isoformat(),
        "target": 90,
        "battery": 10,
        "predicted_level_at_alarm": 50,
    }

    results = {"DeviceX": pd}

    coordinator._handle_post_alarm_self_heal(results, now)

    assert "switch.unit_test" in coordinator._post_alarm_last_handled
    assert any(
        c.get("entity") == "switch.unit_test"
        for c in coordinator._post_alarm_corrections
    )


async def test_post_alarm_edgecases_and_persistence(hass):
    # Reuse constructs from original edgecase tests to validate behavior
    async def _make_coord(hass, entry):
        entry.add_to_hass(hass)
        coord = SmartChargerCoordinator(hass, entry)
        hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
            "entry": entry,
            "coordinator": coord,
            "learning": None,
            "state_machine": None,
        }
        return coord

    # No alarm time is ignored
    device = {
        "name": "PersistDevice",
        "battery_sensor": "sensor.x",
        "charger_switch": "switch.persist",
        "target_level": 90,
    }
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})
    entry.add_to_hass(hass)
    coordinator = SmartChargerCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
        "entry": entry,
        "coordinator": coordinator,
        "learning": None,
        "state_machine": None,
    }

    now = dt_util.utcnow()
    pd = {
        "charger_switch": "switch.no_alarm",
        "alarm_time": None,
        "target": 90,
        "battery": 10,
    }
    coordinator._handle_post_alarm_self_heal({"D": pd}, now)
    assert "switch.no_alarm" not in coordinator._post_alarm_last_handled

    # Future alarm ignored
    entry2 = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    coord2 = await _make_coord(hass, entry2)
    future = dt_util.utcnow() + timedelta(hours=1)
    pd2 = {
        "charger_switch": "switch.future",
        "alarm_time": dt_util.as_local(future).isoformat(),
        "target": 90,
        "battery": 10,
    }
    coord2._handle_post_alarm_self_heal({"D": pd2}, now)
    assert "switch.future" not in coord2._post_alarm_last_handled

    # Battery at or above target marks handled
    entry3 = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    coord3 = await _make_coord(hass, entry3)
    past = dt_util.utcnow() - timedelta(minutes=2)
    pd3 = {
        "charger_switch": "switch.at_target",
        "alarm_time": dt_util.as_local(past).isoformat(),
        "target": 50,
        "battery": 50,
    }
    coord3._handle_post_alarm_self_heal({"D": pd3}, now)
    assert "switch.at_target" in coord3._post_alarm_last_handled


async def test_post_alarm_self_heal_flows(hass):
    # This test exercises the larger post-alarm self-heal flows including
    # persistence and learning reset scheduling. It mirrors earlier integration
    # tests but keeps everything in one grouped test for simplicity.
    device = {
        "name": "D1",
        "battery_sensor": "sensor.bat1",
        "charger_switch": "switch.d1",
        "target_level": 95,
    }
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": [device]}, options={})

    async def _make_coordinator_with_entry(hass, entry):
        entry.add_to_hass(hass)
        coordinator = SmartChargerCoordinator(hass, entry)
        hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = {
            "entry": entry,
            "coordinator": coordinator,
            "learning": type(
                "L",
                (),
                {"async_reset_profile": (lambda *_args, **_kwargs: asyncio.sleep(0))},
            )(),
            "state_machine": None,
        }
        return coordinator

    coordinator = await _make_coordinator_with_entry(hass, entry)

    now = dt_util.utcnow()
    alarm = now - timedelta(minutes=1)
    pd = {
        "charger_switch": "switch.d1",
        "alarm_time": dt_util.as_local(alarm).isoformat(),
        "target": 95,
        "battery": 50,
        "predicted_level_at_alarm": 80,
    }
    epoch = (now - timedelta(seconds=10)).timestamp()
    coordinator._flipflop_events["switch.d1"] = [epoch, epoch + 1, epoch + 2, epoch + 3]
    coordinator._flipflop_warn_threshold = 3

    # Monkeypatch coordinator._build_plan
    from custom_components.smart_charger.coordinator import SmartChargePlan

    async def fake_build_plan(device, now_local, learning, learning_window_hours):
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

    # Spy on config_entries.async_update_entry
    called = []
    orig_update = hass.config_entries.async_update_entry

    def spy_update(entry_obj, **kwargs):
        opts = dict(getattr(entry_obj, "options", {}) or {})
        opts.update(kwargs.get("options", {}) or {})
        entry_obj.options = opts
        called.append(kwargs.get("options", {}))

    hass.config_entries.async_update_entry = spy_update

    # run refresh twice
    await coordinator.async_refresh()
    await asyncio.sleep(0)
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=5)).isoformat()
    await coordinator.async_refresh()
    await asyncio.sleep(0)

    hass.config_entries.async_update_entry = orig_update

    evidence = bool(
        called
        or coordinator._post_alarm_temp_overrides
        or coordinator._adaptive_throttle_overrides
        or coordinator._post_alarm_miss_streaks
        or coordinator._post_alarm_corrections
        or coordinator._post_alarm_last_handled
    )
    assert evidence
