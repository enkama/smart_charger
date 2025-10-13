from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from .const import (
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_CHARGING_SENSOR,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_PRECHARGE_MARGIN_ON,
    CONF_PRESENCE_SENSOR,
    CONF_SMART_START_MARGIN,
    DEFAULT_PRECHARGE_MARGIN_OFF,
    DEFAULT_PRECHARGE_MARGIN_ON,
    DEFAULT_SMART_START_MARGIN,
    DOMAIN,
)
from .learning import SESSION_RETRY_DELAYS

_LOGGER = logging.getLogger(__name__)


def _entity_domain(entity_id: Any) -> str:
    """Extracts the domain part of an entity_id ('sensor.xxx' -> 'sensor')."""
    if not isinstance(entity_id, str) or "." not in entity_id:
        return "<unknown>"
    return entity_id.split(".", 1)[0]


def _extract_state(state_obj: Optional[State]) -> Dict[str, Any]:
    """Return sanitized state and key attributes."""
    if not state_obj:
        return {"state": None, "attributes": {}}
    return {
        "state": state_obj.state,
        "attributes": {
            k: v
            for k, v in state_obj.attributes.items()
            if k in ("unit_of_measurement", "device_class", "charging_state")
        },
    }


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _resolved_margin(value: Optional[float], default: float) -> float:
    if value is None or value < 0:
        return default
    return value


def _coordinator_meta(coordinator) -> Dict[str, Any]:
    if not coordinator:
        return {}

    update_interval = getattr(coordinator, "update_interval", None)
    interval_seconds = update_interval.total_seconds() if update_interval else None

    last_success: Optional[datetime] = getattr(
        coordinator, "_last_successful_update", None
    )
    last_success_iso = (
        last_success.isoformat() if isinstance(last_success, datetime) else None
    )
    elapsed_since_success: Optional[float] = None
    if last_success:
        elapsed_since_success = (dt_util.utcnow() - last_success).total_seconds()

    next_refresh_eta: Optional[float] = None
    if interval_seconds is not None and elapsed_since_success is not None:
        next_refresh_eta = max(0.0, interval_seconds - elapsed_since_success)

    pending_refresh = getattr(coordinator, "_refresh_pending", False)

    return {
        "update_interval_seconds": interval_seconds,
        "last_successful_update": last_success_iso,
        "seconds_since_last_success": elapsed_since_success,
        "next_refresh_eta_seconds": next_refresh_eta,
        "refresh_pending": pending_refresh,
    }


def _summarize_error_history(state_machine) -> Dict[str, Any]:
    if not state_machine or not getattr(state_machine, "error_history", None):
        return {"recent": [], "counts": {}}

    recent: list[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for key, timestamps in state_machine.error_history.items():
        try:
            profile_id, error = key.split(":", 1)
        except ValueError:
            continue

        count = len(timestamps)
        last_ts = timestamps[-1] if timestamps else None
        counts[error] = counts.get(error, 0) + count
        recent.append(
            {
                "profile_id": profile_id,
                "error": error,
                "count": count,
                "last_occurrence": last_ts,
            }
        )

    recent.sort(key=lambda item: item.get("last_occurrence") or "", reverse=True)
    return {"recent": recent[:10], "counts": counts}


def _collect_device_diagnostics(
    hass: HomeAssistant, devices: list[Dict[str, Any]]
) -> list[Dict[str, Any]]:
    sanitized_devices: list[Dict[str, Any]] = []

    for dev in devices:
        name = dev.get("name", "<unnamed>")
        batt_ent = dev.get(CONF_BATTERY_SENSOR)
        charger_ent = dev.get(CONF_CHARGER_SWITCH)
        charging_ent = dev.get(CONF_CHARGING_SENSOR)
        presence_ent = dev.get(CONF_PRESENCE_SENSOR)
        margin_on = _optional_float(dev.get(CONF_PRECHARGE_MARGIN_ON))
        margin_off = _optional_float(dev.get(CONF_PRECHARGE_MARGIN_OFF))
        smart_margin = _optional_float(dev.get(CONF_SMART_START_MARGIN))

        sanitized_devices.append(
            {
                "name": name,
                "battery_sensor": {
                    "id": batt_ent,
                    "domain": _entity_domain(batt_ent),
                    "state": (
                        _extract_state(hass.states.get(str(batt_ent)))
                        if batt_ent
                        else {}
                    ),
                },
                "charger_switch": {
                    "id": charger_ent,
                    "domain": _entity_domain(charger_ent),
                    "state": (
                        _extract_state(hass.states.get(str(charger_ent)))
                        if charger_ent
                        else {}
                    ),
                },
                "charging_sensor": {
                    "id": charging_ent,
                    "domain": _entity_domain(charging_ent),
                    "state": (
                        _extract_state(hass.states.get(str(charging_ent)))
                        if charging_ent
                        else {}
                    ),
                },
                "presence_sensor": {
                    "id": presence_ent,
                    "domain": _entity_domain(presence_ent),
                    "state": (
                        _extract_state(hass.states.get(str(presence_ent)))
                        if presence_ent
                        else {}
                    ),
                },
                "target_level": dev.get("target_level"),
                "min_level": dev.get("min_level"),
                "precharge_level": dev.get("precharge_level"),
                "precharge_margin_on": margin_on,
                "precharge_margin_off": margin_off,
                "smart_start_margin": smart_margin,
                "effective_margins": {
                    "precharge_margin_on": _resolved_margin(
                        margin_on, DEFAULT_PRECHARGE_MARGIN_ON
                    ),
                    "precharge_margin_off": _resolved_margin(
                        margin_off, DEFAULT_PRECHARGE_MARGIN_OFF
                    ),
                    "smart_start_margin": _resolved_margin(
                        smart_margin, DEFAULT_SMART_START_MARGIN
                    ),
                },
            }
        )

    return sanitized_devices


def _build_learning_summary(learning) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if learning and getattr(learning, "_data", None):
        profiles = getattr(learning, "_data", {})
        summary["profile_count"] = len(profiles)
        summary["profiles"] = list(profiles.keys())
        summary["stats"] = {
            pid: {
                "sample_count": len(pdata.get("samples", [])),
                "cycle_count": len(pdata.get("cycles", [])),
                "last_cycle": (pdata.get("cycles") or [None])[-1],
                "ema_avg_speed": (pdata.get("stats", {}) or {}).get("ema"),
                "bucket_stats": pdata.get("bucket_stats", {}),
            }
            for pid, pdata in profiles.items()
        }
        active_sessions = {}
        for pid, pdata in profiles.items():
            session = pdata.get("current_session")
            if session:
                active_sessions[pid] = {
                    "started": session.get("started"),
                    "sensor": session.get("sensor"),
                    "retries": session.get("retries", 0),
                }
        if active_sessions:
            summary["active_sessions"] = active_sessions
        summary["retry_schedule_seconds"] = list(SESSION_RETRY_DELAYS)

    return summary


def _build_error_heatmaps(state_machine) -> tuple[Dict[str, Any], Dict[str, Any]]:
    error_heatmap: Dict[str, Any] = {}
    global_heatmap: Dict[str, Any] = {}

    if state_machine and getattr(state_machine, "error_history", None):
        for key, timestamps in state_machine.error_history.items():
            try:
                pid, etype = key.split(":", 1)
            except ValueError:
                continue
            if pid not in error_heatmap:
                error_heatmap[pid] = {}
            if etype not in error_heatmap[pid]:
                error_heatmap[pid][etype] = [0] * 24
            if etype not in global_heatmap:
                global_heatmap[etype] = [0] * 24

            for ts in timestamps:
                dt = None
                try:
                    dt = dt_util.parse_datetime(ts)
                except (ValueError, TypeError) as err:
                    _LOGGER.debug("Failed to parse error timestamp %s: %s", ts, err)
                if not dt:
                    continue
                hour = dt.hour
                error_heatmap[pid][etype][hour] += 1
                global_heatmap[etype][hour] += 1

    return error_heatmap, global_heatmap


def _capture_coordinator_state(
    coordinator,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    coordinator_state: Dict[str, Any] = {}
    coordinator_plans: Dict[str, Any] = {}
    coordinator_insights: Dict[str, Any] = {}

    if coordinator and getattr(coordinator, "profiles", None):
        now_utc = dt_util.utcnow()
        precharge_active = []
        smart_active = []
        release_map = dict(getattr(coordinator, "_precharge_release", {}) or {})

        for pid, plan in coordinator.profiles.items():
            plan_copy = dict(plan)
            alarm_iso = plan_copy.get("alarm_time")
            next_event_delta: Optional[float] = None
            if alarm_iso:
                try:
                    alarm_dt = dt_util.parse_datetime(alarm_iso)
                    if alarm_dt:
                        next_event_delta = (alarm_dt - now_utc).total_seconds()
                except Exception:
                    next_event_delta = None

            plan_copy["seconds_until_alarm"] = next_event_delta
            coordinator_plans[pid] = plan_copy

            coordinator_state[pid] = {
                "battery": plan_copy.get("battery"),
                "target": plan_copy.get("target"),
                "avg_speed": plan_copy.get("avg_speed"),
                "charging_state": plan_copy.get("charging_state"),
                "presence_state": plan_copy.get("presence_state"),
                "duration_min": plan_copy.get("duration_min"),
                "start_time": plan_copy.get("start_time"),
                "alarm_time": plan_copy.get("alarm_time"),
                "seconds_until_alarm": next_event_delta,
                "precharge_level": plan_copy.get("precharge_level"),
                "precharge_margin_on": plan_copy.get("precharge_margin_on"),
                "precharge_margin_off": plan_copy.get("precharge_margin_off"),
                "smart_start_margin": plan_copy.get("smart_start_margin"),
                "precharge_release_level": release_map.get(pid),
                "precharge_active": plan_copy.get("precharge_active"),
                "smart_start_active": plan_copy.get("smart_start_active"),
                "predicted_level_at_alarm": plan_copy.get("predicted_level_at_alarm"),
                "predicted_drain": plan_copy.get("predicted_drain"),
                "last_update": plan_copy.get("last_update"),
            }

            if plan_copy.get("precharge_active"):
                precharge_active.append(pid)
            if plan_copy.get("smart_start_active"):
                smart_active.append(pid)

        coordinator_insights = {
            "active_profiles": list(coordinator_state.keys()),
            "precharge_active": precharge_active,
            "smart_start_active": smart_active,
            "precharge_release_levels": release_map,
        }

    return coordinator_state, coordinator_plans, coordinator_insights


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> Dict[str, Any]:
    """Return diagnostics for a Smart Charger config entry."""
    domain_data = hass.data[DOMAIN]
    data = domain_data["entries"].get(entry.entry_id, {})
    state_machine = data.get("state_machine")
    coordinator = data.get("coordinator")
    learning = data.get("learning")
    devices = entry.data.get("devices") or []
    sanitized_devices = _collect_device_diagnostics(hass, devices)
    learning_summary = _build_learning_summary(learning)
    error_heatmap, global_heatmap = _build_error_heatmaps(state_machine)
    coordinator_state, coordinator_plans, coordinator_insights = (
        _capture_coordinator_state(coordinator)
    )
    coordinator_meta = _coordinator_meta(coordinator)
    state_machine_summary = state_machine.as_dict() if state_machine else {}
    error_summary = _summarize_error_history(state_machine)

    return {
        "entry_data": {"devices": sanitized_devices},
        "options": getattr(entry, "options", {}),
        "coordinator_profiles": coordinator_state,
        "coordinator_plans": coordinator_plans,
        "coordinator_meta": coordinator_meta,
        "coordinator_insights": coordinator_insights,
        "learning_summary": learning_summary,
        "state_machine": state_machine_summary,
        "state_machine_errors": error_summary,
        "suggestions": (
            state_machine.get_suggestions()
            if state_machine and hasattr(state_machine, "get_suggestions")
            else []
        ),
        "error_heatmap": error_heatmap,
        "global_error_heatmap": global_heatmap,
        "last_update": dt_util.now().isoformat(),
    }
