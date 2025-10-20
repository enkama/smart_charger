from __future__ import annotations

import logging
from typing import Any, Dict

from homeassistant.components.sensor import SensorEntity
from homeassistant.const import STATE_UNKNOWN
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.util import dt as dt_util

from .const import DOMAIN, CONF_ADAPTIVE_EWMA_ALPHA, DEFAULT_ADAPTIVE_EWMA_ALPHA

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up Smart Charger sensors."""
    domain_data = hass.data[DOMAIN]
    data = domain_data["entries"][entry.entry_id]
    coordinator = data["coordinator"]
    state_machine = data["state_machine"]
    learning = data["learning"]

    devices = entry.data.get("devices", [])
    entities: list[SensorEntity] = [
        SmartChargerNextStartSensor(coordinator, state_machine),
        SmartChargerLearningSensor(learning),
    ]

    for device in devices:
        name = device.get("name")
        if name:
            entities.append(SmartChargerDeviceSensor(name, coordinator))

    # Add adaptive telemetry sensor (global across the integration)
    entities.append(SmartChargerAdaptiveSensor(coordinator))

    async_add_entities(entities, True)


class SmartChargerNextStartSensor(SensorEntity):
    """Provide the earliest scheduled start across all devices."""

    _attr_name = "Smart Charger Next Start"
    _attr_icon = "mdi:battery-clock"
    _attr_should_poll = False
    _attr_translation_key = "next_start"

    def __init__(self, coordinator, state_machine) -> None:
        self.coordinator = coordinator
        self.state_machine = state_machine
        self._attr_unique_id = f"{DOMAIN}_next_start"
        self._attr_native_value = STATE_UNKNOWN
        self._attr_extra_state_attributes: Dict[str, Any] = {
            "info": "No profiles available"
        }

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if hasattr(self.coordinator, "async_add_listener"):
            self.async_on_remove(
                self.coordinator.async_add_listener(self._handle_coordinator_update)
            )
        self._handle_coordinator_update()
        await self.coordinator.async_request_refresh()

    def _handle_coordinator_update(self) -> None:
        profiles = self.coordinator.profiles or {}
        sm = self.state_machine

        if not profiles:
            self._attr_native_value = STATE_UNKNOWN
            self._attr_extra_state_attributes = {"info": "No profiles available"}
            self.async_write_ha_state()
            return

        start_times: list[str] = []
        for plan in profiles.values():
            start_time = plan.get("start_time")
            if start_time and not plan.get("skipped"):
                start_times.append(str(start_time))
        self._attr_native_value = min(start_times) if start_times else STATE_UNKNOWN

        device_data: Dict[str, Dict[str, Any]] = {}
        for pid, data in profiles.items():
            device_data[pid] = {
                "battery": data.get("battery"),
                "target": data.get("target"),
                "avg_speed": data.get("avg_speed"),
                "duration_min": data.get("duration_min"),
                "charge_duration_min": data.get("charge_duration_min"),
                "total_duration_min": data.get("total_duration_min"),
                "precharge_duration_min": data.get("precharge_duration_min"),
                "alarm_time": data.get("alarm_time"),
                "start_time": data.get("start_time"),
                "predicted_drain": data.get("predicted_drain"),
                "predicted_level_at_alarm": data.get("predicted_level_at_alarm"),
                "drain_rate": data.get("drain_rate"),
                "drain_confidence": data.get("drain_confidence"),
                "drain_basis": data.get("drain_basis"),
                "smart_start_active": data.get("smart_start_active"),
                "precharge_level": data.get("precharge_level"),
                "precharge_active": data.get("precharge_active"),
                "precharge_release_level": data.get("precharge_release_level"),
                "charging_state": data.get("charging_state"),
                "presence_state": data.get("presence_state", "unknown"),
                "skipped": data.get("skipped", False),
                "last_update": data.get("last_update"),
            }
            for key in (
                "charge_duration_min",
                "total_duration_min",
                "precharge_duration_min",
            ):
                if device_data[pid].get(key) is None:
                    device_data[pid].pop(key, None)

        extra: Dict[str, Any] = {
            "devices": device_data,
            "device_count": len(device_data),
            "smart_start_active_count": sum(
                1 for d in device_data.values() if d.get("smart_start_active")
            ),
            "last_update": dt_util.now().isoformat(),
            "state_machine": sm.as_dict() if sm else {},
        }
        self._attr_extra_state_attributes = extra
        self.async_write_ha_state()


class SmartChargerLearningSensor(SensorEntity):
    """Report aggregate learning statistics for the integration."""

    _attr_name = "Smart Charger Learning Stats"
    _attr_icon = "mdi:chart-line"
    _attr_should_poll = False
    _attr_translation_key = "learning_stats"

    def __init__(self, learning) -> None:
        self.learning = learning
        self._attr_unique_id = f"{DOMAIN}_learning_stats"
        self._attr_native_value = 1.0
        self._attr_extra_state_attributes: Dict[str, Any] = {
            "profile_count": 0,
            "sample_count": 0,
            "profiles": [],
            "last_update": dt_util.now().isoformat(),
        }

    async def async_update(self) -> None:
        try:
            self._attr_native_value = float(self.learning.avg_speed())
        except Exception:
            self._attr_native_value = 1.0

        snapshot = self.learning.snapshot()
        profiles = snapshot.get("profiles", {})
        meta = snapshot.get("meta", {})
        sample_count = meta.get("sample_count")
        if sample_count is None:
            sample_count = sum(len(p.get("samples", [])) for p in profiles.values())
        cycle_count = meta.get("cycle_count")
        if cycle_count is None:
            cycle_count = sum(len(p.get("cycles", [])) for p in profiles.values())
        extra: Dict[str, Any] = {
            "profile_count": meta.get("profile_count", len(profiles)),
            "sample_count": sample_count,
            "cycle_count": cycle_count,
            "model_revision": meta.get("model_revision"),
            "profiles": list(profiles.keys()),
            "last_update": dt_util.now().isoformat(),
        }
        self._attr_extra_state_attributes = extra

        _LOGGER.debug("Smart Charger Learning Sensor: manual refresh completed.")


class SmartChargerDeviceSensor(SensorEntity):
    """Expose per-device status with a dynamic icon."""

    _attr_should_poll = False

    def __init__(self, device_name: str, coordinator) -> None:
        self.coordinator = coordinator
        self.device_name = device_name
        self._attr_name = f"Smart Charger {device_name} Status"
        self._attr_unique_id = (
            f"{DOMAIN}_{device_name.lower().replace(' ', '_')}_status"
        )
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device_name.lower().replace(" ", "_"))},
            name=f"Smart Charger – {device_name}",
            manufacturer="Smart Charger System",
            model="Predictive Charging v2",
            configuration_url=("https://my.home-assistant.io/redirect/integrations/"),
        )
        self._attr_native_value = STATE_UNKNOWN
        self._attr_extra_state_attributes: Dict[str, Any] = {
            "info": "No data available"
        }
        self._attr_icon = "mdi:battery-question"
        self._attr_translation_key = "device_status"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if hasattr(self.coordinator, "async_add_listener"):
            self.async_on_remove(
                self.coordinator.async_add_listener(self._handle_coordinator_update)
            )
        self._handle_coordinator_update()
        await self.coordinator.async_request_refresh()

    def _handle_coordinator_update(self) -> None:
        profiles = self.coordinator.profiles or {}
        data = profiles.get(self.device_name)
        prev_attrs = dict(self._attr_extra_state_attributes or {})
        if not data:
            fallback_attrs = {"info": "No data available"}
            if (
                self._attr_native_value != STATE_UNKNOWN
                or self._attr_icon != "mdi:battery-question"
                or prev_attrs != fallback_attrs
            ):
                self._attr_native_value = STATE_UNKNOWN
                self._attr_icon = "mdi:battery-question"
                self._attr_extra_state_attributes = fallback_attrs
                self.async_write_ha_state()
            return

        if data.get("skipped"):
            status = "skipped"
        elif data.get("precharge_active"):
            status = "precharging"
        elif data.get("charging_state") == "charging":
            status = "charging"
        elif data.get("charging_state") == "full":
            status = "full"
        elif data.get("smart_start_active"):
            status = "scheduled"
        else:
            status = "waiting"

        icon_map = {
            "charging": "mdi:battery-charging",
            "full": "mdi:battery",
            "scheduled": "mdi:clock-start",
            "waiting": "mdi:clock-outline",
            "precharging": "mdi:battery-plus",
            "skipped": "mdi:battery-off",
        }

        self._attr_native_value = status
        self._attr_icon = icon_map.get(status, "mdi:battery")
        extra: Dict[str, Any] = {
            "battery": data.get("battery"),
            "target": data.get("target"),
            "avg_speed": data.get("avg_speed"),
            "duration_min": data.get("duration_min"),
            "charge_duration_min": data.get("charge_duration_min"),
            "total_duration_min": data.get("total_duration_min"),
            "precharge_duration_min": data.get("precharge_duration_min"),
            "alarm_time": data.get("alarm_time"),
            "start_time": data.get("start_time"),
            "predicted_drain": data.get("predicted_drain"),
            "predicted_level_at_alarm": data.get("predicted_level_at_alarm"),
            "drain_rate": data.get("drain_rate"),
            "drain_confidence": data.get("drain_confidence"),
            "drain_basis": data.get("drain_basis"),
            "smart_start_active": data.get("smart_start_active"),
            "precharge_level": data.get("precharge_level"),
            "precharge_active": data.get("precharge_active"),
            "precharge_release_level": data.get("precharge_release_level"),
            "charging_state": data.get("charging_state"),
            "presence_state": data.get("presence_state", "unknown"),
            "skipped": data.get("skipped", False),
            "last_update": data.get("last_update"),
        }
        for key in (
            "charge_duration_min",
            "total_duration_min",
            "precharge_duration_min",
        ):
            if extra.get(key) is None:
                extra.pop(key, None)
        if (
            self._attr_native_value != status
            or self._attr_icon != icon_map.get(status, "mdi:battery")
            or prev_attrs != extra
        ):
            self._attr_native_value = status
            self._attr_icon = icon_map.get(status, "mdi:battery")
            self._attr_extra_state_attributes = extra
            self.async_write_ha_state()

    async def async_update(self) -> None:
        await self.coordinator.async_request_refresh()


class SmartChargerAdaptiveSensor(SensorEntity):
    """Expose integration-level adaptive telemetry: flip-flop counts and active overrides."""

    _attr_name = "Smart Charger Adaptive Telemetry"
    _attr_icon = "mdi:shield-alert"
    _attr_should_poll = False
    _attr_translation_key = "adaptive_telemetry"

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator
        self._attr_unique_id = f"{DOMAIN}_adaptive_telemetry"
        self._attr_native_value = 0
        self._attr_extra_state_attributes: Dict[str, Any] = {
            "flipflop_counts": {},
            "active_overrides": {},
            "last_update": None,
        }

        # Initialize coordinator-side EWMA state if missing
        if not hasattr(self.coordinator, "_flipflop_ewma"):
            # store as events/sec EWMA
            self.coordinator._flipflop_ewma = 0.0

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if hasattr(self.coordinator, "async_add_listener"):
            self.async_on_remove(
                self.coordinator.async_add_listener(self._handle_coordinator_update)
            )
        self._handle_coordinator_update()
        await self.coordinator.async_request_refresh()

    def _handle_coordinator_update(self) -> None:
        # flipflop_events is a dict: entity -> list[timestamps]
        ff = getattr(self.coordinator, "_flipflop_events", {}) or {}
        counts = {k: len(v) for k, v in ff.items()}

        overrides = getattr(self.coordinator, "_adaptive_throttle_overrides", {}) or {}
        active = {
            ent: {"applied": ov.get("applied"), "expires": ov.get("expires")}
            for ent, ov in overrides.items()
        }

        total_events = sum(counts.values())

        # compute window size from coordinator (fallback to 300s)
        window = getattr(self.coordinator, "_flipflop_window_seconds", None)
        if window is None:
            window = getattr(self.coordinator, "_flipflop_window_seconds", 300.0)

        # events per second and per minute
        rate_per_sec = total_events / float(window) if window > 0 else 0.0
        rate_per_min = rate_per_sec * 60.0

        # determine EWMA alpha from config options (fallback to default)
        entry_obj = getattr(self.coordinator, "entry", None)
        try:
            if entry_obj and getattr(entry_obj, "options", None) is not None:
                alpha = float(entry_obj.options.get(CONF_ADAPTIVE_EWMA_ALPHA, DEFAULT_ADAPTIVE_EWMA_ALPHA))
            else:
                alpha = DEFAULT_ADAPTIVE_EWMA_ALPHA
        except Exception:
            _LOGGER.debug("Failed to read EWMA alpha from entry options, using default", exc_info=True)
            alpha = DEFAULT_ADAPTIVE_EWMA_ALPHA

        # compute and persist EWMA (events/sec)
        prev = getattr(self.coordinator, "_flipflop_ewma", 0.0)
        try:
            ewma = prev + alpha * (rate_per_sec - prev)
        except Exception:
            _LOGGER.debug("Failed to compute EWMA, falling back to previous value", exc_info=True)
            ewma = prev
        self.coordinator._flipflop_ewma = ewma

        self._attr_native_value = total_events
        # expose coordinator EWMA metadata when available
        ewma_last = getattr(self.coordinator, "_flipflop_ewma_last_update", None)
        ewma_exceeded = bool(getattr(self.coordinator, "_flipflop_ewma_exceeded", False))
        self._attr_extra_state_attributes = {
            "flipflop_counts": counts,
            "flipflop_rate_per_second": round(rate_per_sec, 6),
            "flipflop_rate_per_minute": round(rate_per_min, 3),
            "flipflop_ewma_per_second": round(ewma, 6),
            "flipflop_ewma_last_update": dt_util.as_local(dt_util.utc_from_timestamp(ewma_last)).isoformat() if ewma_last else None,
            "flipflop_ewma_exceeded": ewma_exceeded,
            "active_overrides": active,
            "active_override_count": len(active),
            "last_update": dt_util.now().isoformat(),
        }
        self.async_write_ha_state()

    async def async_update(self) -> None:
        await self.coordinator.async_request_refresh()
