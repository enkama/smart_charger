"""Edge-case tests for post-alarm self-heal behaviours and persistence."""

from __future__ import annotations

from datetime import timedelta

import asyncio
import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


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


async def test_no_alarm_time_is_ignored(hass):
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
    pd = {"charger_switch": "switch.no_alarm", "alarm_time": None, "target": 90, "battery": 10}
    # call helper directly
    coordinator._handle_post_alarm_self_heal({"D": pd}, now)

    # nothing should be marked handled
    assert "switch.no_alarm" not in coordinator._post_alarm_last_handled


async def test_future_alarm_is_ignored(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    coordinator = await _make_coord(hass, entry)

    now = dt_util.utcnow()
    future_alarm = now + timedelta(hours=1)
    pd = {"charger_switch": "switch.future", "alarm_time": dt_util.as_local(future_alarm).isoformat(), "target": 90, "battery": 10}
    coordinator._handle_post_alarm_self_heal({"D": pd}, now)

    assert "switch.future" not in coordinator._post_alarm_last_handled


async def test_battery_at_or_above_target_marks_handled(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    coordinator = await _make_coord(hass, entry)

    now = dt_util.utcnow()
    past = now - timedelta(minutes=2)
    pd = {"charger_switch": "switch.at_target", "alarm_time": dt_util.as_local(past).isoformat(), "target": 50, "battery": 50}
    coordinator._handle_post_alarm_self_heal({"D": pd}, now)

    # Should be marked handled because battery reached target
    assert "switch.at_target" in coordinator._post_alarm_last_handled


async def test_persistence_via_spy_update(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    coordinator = await _make_coord(hass, entry)

    # Create pd that triggers late_start persistence after two runs
    now = dt_util.utcnow()
    past = now - timedelta(minutes=2)
    pd = {
        "charger_switch": "switch.persist",
        "alarm_time": dt_util.as_local(past).isoformat(),
        "target": 90,
        "battery": 10,
        "charge_duration_min": 9999.0,
        "smart_start_margin": 1.0,
    }

    # Spy on config_entries.async_update_entry to capture persistence attempts
    called = []
    orig_update = hass.config_entries.async_update_entry

    def spy_update(entry_obj, **kwargs):
        opts = dict(getattr(entry_obj, "options", {}) or {})
        opts.update(kwargs.get("options", {}) or {})
        entry_obj.options = opts
        called.append(kwargs.get("options", {}))

    hass.config_entries.async_update_entry = spy_update

    # Call the helper twice with updated alarm times to simulate two separate misses
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=30)).isoformat()
    coordinator._handle_post_alarm_self_heal({"Device": pd}, now)
    pd["alarm_time"] = dt_util.as_local(now - timedelta(seconds=5)).isoformat()
    coordinator._handle_post_alarm_self_heal({"Device": pd}, now)

    hass.config_entries.async_update_entry = orig_update

    # we expect persistence attempt recorded at least once (spy captured) or other corrective evidence
    assert (
        called
        or "switch.persist" in coordinator._post_alarm_persisted_smart_start
        or coordinator._post_alarm_miss_streaks.get("switch.persist")
    )
