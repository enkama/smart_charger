from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Dict, Iterable, Mapping, Optional

from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    ALARM_MODE_PER_DAY,
    ALARM_MODE_SINGLE,
    CHARGING_STATES,
    CONF_ALARM_ENTITY,
    CONF_ALARM_FRIDAY,
    CONF_ALARM_MODE,
    CONF_ALARM_MONDAY,
    CONF_ALARM_SATURDAY,
    CONF_ALARM_SUNDAY,
    CONF_ALARM_THURSDAY,
    CONF_ALARM_TUESDAY,
    CONF_ALARM_WEDNESDAY,
    CONF_AVG_SPEED_SENSOR,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_CHARGING_SENSOR,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_PRECHARGE_MARGIN_ON,
    CONF_PRECHARGE_COUNTDOWN_WINDOW,
    CONF_LEARNING_RECENT_SAMPLE_HOURS,
    CONF_SWITCH_THROTTLE_SECONDS,
    CONF_SWITCH_CONFIRMATION_COUNT,
    CONF_PRESENCE_SENSOR,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    CONF_SMART_START_MARGIN,
    DEFAULT_TARGET_LEVEL,
    DEFAULT_PRECHARGE_MARGIN_OFF,
    DEFAULT_PRECHARGE_MARGIN_ON,
    DEFAULT_SMART_START_MARGIN,
    DEFAULT_PRECHARGE_COUNTDOWN_WINDOW,
    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    DEFAULT_SWITCH_THROTTLE_SECONDS,
    DEFAULT_SWITCH_CONFIRMATION_COUNT,
    DISCHARGING_STATES,
    DOMAIN,
    DEFAULT_FALLBACK_MINUTES_PER_PERCENT,
    LEARNING_DEFAULT_SPEED,
    LEARNING_MAX_SPEED,
    LEARNING_MIN_SPEED,
    FULL_STATES,
    MAX_OBSERVED_DRAIN_RATE,
    UNKNOWN_STATES,
    UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

PRECHARGE_RELEASE_HYSTERESIS = timedelta(minutes=10)

WEEKDAY_ALARM_FIELDS: tuple[str, ...] = (
    CONF_ALARM_MONDAY,
    CONF_ALARM_TUESDAY,
    CONF_ALARM_WEDNESDAY,
    CONF_ALARM_THURSDAY,
    CONF_ALARM_FRIDAY,
    CONF_ALARM_SATURDAY,
    CONF_ALARM_SUNDAY,
)


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    battery_sensor: str
    charger_switch: str
    target_level: float
    min_level: float
    precharge_level: float
    use_predictive_mode: bool
    precharge_margin_on: Optional[float] = None
    precharge_margin_off: Optional[float] = None
    smart_start_margin: Optional[float] = None
    switch_throttle_seconds: Optional[float] = None
    switch_confirmation_count: Optional[int] = None
    charging_sensor: Optional[str] = None
    avg_speed_sensor: Optional[str] = None
    presence_sensor: Optional[str] = None
    alarm_mode: str = ALARM_MODE_SINGLE
    alarm_entity: Optional[str] = None
    learning_recent_sample_hours: Optional[float] = None
    alarm_entities_by_weekday: Mapping[int, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DeviceConfig":
        name = str(raw.get("name", "")).strip()
        battery_sensor = raw.get(CONF_BATTERY_SENSOR)
        charger_switch = raw.get(CONF_CHARGER_SWITCH)

        if not name or not battery_sensor or not charger_switch:
            raise ValueError(f"Incomplete device configuration: {raw}")

        alarm_mode = str(raw.get(CONF_ALARM_MODE, ALARM_MODE_SINGLE))
        weekday_map: dict[int, str] = {}
        for idx, field_name in enumerate(WEEKDAY_ALARM_FIELDS):
            ent = raw.get(field_name)
            if ent:
                weekday_map[idx] = ent

        def _coerce_margin(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return max(0.0, parsed)

        def _coerce_learning_window(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            clamped = max(0.25, min(48.0, parsed))
            return clamped

        def _coerce_confirmation(value: Any) -> Optional[int]:
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return max(1, min(10, parsed))

        return cls(
            name=name,
            battery_sensor=battery_sensor,
            charger_switch=charger_switch,
            charging_sensor=raw.get(CONF_CHARGING_SENSOR),
            avg_speed_sensor=raw.get(CONF_AVG_SPEED_SENSOR),
            presence_sensor=raw.get(CONF_PRESENCE_SENSOR),
            target_level=float(raw.get(CONF_TARGET_LEVEL, DEFAULT_TARGET_LEVEL)),
            min_level=float(raw.get(CONF_MIN_LEVEL, 30)),
            precharge_level=float(raw.get(CONF_PRECHARGE_LEVEL, 50)),
            use_predictive_mode=bool(raw.get(CONF_USE_PREDICTIVE_MODE, True)),
            precharge_margin_on=_coerce_margin(raw.get(CONF_PRECHARGE_MARGIN_ON)),
            precharge_margin_off=_coerce_margin(raw.get(CONF_PRECHARGE_MARGIN_OFF)),
            smart_start_margin=_coerce_margin(raw.get(CONF_SMART_START_MARGIN)),
            alarm_mode=alarm_mode,
            alarm_entity=raw.get(CONF_ALARM_ENTITY),
            switch_throttle_seconds=_coerce_learning_window(
                raw.get(CONF_SWITCH_THROTTLE_SECONDS)
            ),
            switch_confirmation_count=_coerce_confirmation(
                raw.get(CONF_SWITCH_CONFIRMATION_COUNT)
            ),
            learning_recent_sample_hours=_coerce_learning_window(
                raw.get(CONF_LEARNING_RECENT_SAMPLE_HOURS)
            ),
            alarm_entities_by_weekday=weekday_map,
        )

    def alarm_entity_for_today(self, weekday: int) -> Optional[str]:
        if self.alarm_mode == ALARM_MODE_PER_DAY and self.alarm_entities_by_weekday:
            return self.alarm_entities_by_weekday.get(weekday) or self.alarm_entity
        return self.alarm_entity


@dataclass
class SmartChargePlan:
    battery: float
    target: float
    avg_speed: float
    duration_min: float
    charge_duration_min: float
    total_duration_min: float
    precharge_duration_min: Optional[float]
    alarm_time: datetime
    start_time: Optional[datetime]
    predicted_drain: float
    predicted_level_at_alarm: float
    drain_rate: float
    drain_confidence: float
    drain_basis: tuple[str, ...]
    smart_start_active: bool
    precharge_level: float
    precharge_margin_on: float
    precharge_margin_off: float
    smart_start_margin: float
    precharge_active: bool
    precharge_release_level: Optional[float]
    charging_state: str
    presence_state: str
    last_update: datetime

    def as_dict(self) -> Dict[str, Any]:
        charge_duration_display: Optional[float] = None
        if not math.isclose(self.charge_duration_min, self.duration_min, abs_tol=0.05):
            charge_duration_display = round(self.charge_duration_min, 1)

        total_duration_display: Optional[float] = None
        if not math.isclose(self.total_duration_min, self.duration_min, abs_tol=0.05):
            total_duration_display = round(self.total_duration_min, 1)

        precharge_duration_display: Optional[float] = None
        if self.precharge_duration_min is not None and self.precharge_duration_min > 0:
            precharge_duration_display = round(self.precharge_duration_min, 1)

        return {
            "battery": round(self.battery, 1),
            "target": round(self.target, 1),
            "avg_speed": round(self.avg_speed, 3),
            "duration_min": round(self.duration_min, 1),
            "charge_duration_min": charge_duration_display,
            "total_duration_min": total_duration_display,
            "precharge_duration_min": precharge_duration_display,
            "alarm_time": dt_util.as_local(self.alarm_time).isoformat(),
            "start_time": (
                self._display_start_time()
                if self.start_time
                else None
            ),
            "predicted_drain": round(self.predicted_drain, 2),
            "predicted_level_at_alarm": round(self.predicted_level_at_alarm, 1),
            "drain_rate": round(self.drain_rate, 3),
            "drain_confidence": round(self.drain_confidence, 3),
            "drain_basis": list(self.drain_basis),
            "smart_start_active": self.smart_start_active,
            "precharge_level": round(self.precharge_level, 1),
            "precharge_margin_on": round(self.precharge_margin_on, 2),
            "precharge_margin_off": round(self.precharge_margin_off, 2),
            "smart_start_margin": round(self.smart_start_margin, 2),
            "precharge_active": self.precharge_active,
            "precharge_release_level": (
                round(self.precharge_release_level, 1)
                if self.precharge_release_level is not None
                else None
            ),
            "charging_state": self.charging_state,
            "presence_state": self.presence_state,
            "last_update": dt_util.now().isoformat(),
        }

    def _display_start_time(self) -> str:
        if self.start_time is None:
            raise RuntimeError("SmartChargePlan missing start_time for display")
        local_start = dt_util.as_local(self.start_time)
        now_local = dt_util.as_local(dt_util.utcnow())
        if local_start < now_local:
            local_start = now_local
        return local_start.isoformat()


class SmartChargerCoordinator(DataUpdateCoordinator[Dict[str, Dict[str, Any]]]):
    """Coordinates the calculation and control of the charging plan."""

    _last_action_log: Dict[str, str]

    def __init__(self, hass: HomeAssistant, entry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name="Smart Charger Coordinator",
            update_interval=timedelta(seconds=UPDATE_INTERVAL),
        )
        self.entry = entry
        self.config: Dict[str, Any] = {}
        self._state: Optional[Dict[str, Dict[str, Any]]] = None
        self._last_successful_update: datetime | None = None
        self._last_action_log = {}
        self._precharge_release: Dict[str, float] = {}
        self._precharge_release_ready: Dict[str, datetime] = {}
        self._drain_rate_cache: Dict[str, float] = {}
        self._battery_history: Dict[str, tuple[datetime, float, bool]] = {}
        self._precharge_margin_on = DEFAULT_PRECHARGE_MARGIN_ON
        self._precharge_margin_off = DEFAULT_PRECHARGE_MARGIN_OFF
        self._smart_start_margin = DEFAULT_SMART_START_MARGIN
        self._precharge_countdown_window = DEFAULT_PRECHARGE_COUNTDOWN_WINDOW
        self._default_learning_recent_sample_hours = (
            DEFAULT_LEARNING_RECENT_SAMPLE_HOURS
        )
        # Default throttle (seconds) used when device doesn't specify one
        self._default_switch_throttle_seconds = DEFAULT_SWITCH_THROTTLE_SECONDS
        # Per-device last switch timestamp to avoid rapid on/off flapping
        self._last_switch_time: Dict[str, datetime] = {}
        # Per-device configured throttle values (entity_id -> seconds)
        self._device_switch_throttle: Dict[str, float] = {}
        # Confirmation debounce: require N consecutive coordinator evaluations
        # that request a different desired state before issuing a switch.
        # Default is read from the central constants to remain consistent.
        self._confirmation_required = DEFAULT_SWITCH_CONFIRMATION_COUNT
        # per-device recent desired-state history: tuple(last_desired_bool, count)
        self._desired_state_history: Dict[str, tuple[bool, int]] = {}
        # Per-refresh evaluation id used to avoid double-recording within the
        # same coordinator run when multiple code paths may record the desired
        # state for an entity.
        self._current_eval_id = 0
        self._last_recorded_eval: Dict[str, int] = {}


    @property
    def profiles(self) -> Dict[str, Dict[str, Any]]:
        return self._state or {}

    def _raw_config(self) -> Dict[str, Any]:
        self.config = {**dict(self.entry.data), **getattr(self.entry, "options", {})}
        return self.config

    def _option_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            _LOGGER.debug(
                "Invalid option %s=%s, falling back to %.2f",
                key,
                value,
                default,
            )
            return default
        return max(0.0, parsed)

    def _iter_device_configs(
        self, devices: Optional[Iterable[Mapping[str, Any]]] = None
    ) -> Iterable[DeviceConfig]:
        if devices is None:
            devices = self._raw_config().get("devices") or []
        for raw in list(devices or []):
            try:
                yield DeviceConfig.from_dict(raw)
            except Exception as err:
                _LOGGER.warning(
                    "Skipping invalid device configuration %s: %s", raw, err
                )

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        # New evaluation - advance the eval id to avoid duplicate recordings
        # within the same refresh cycle.
        try:
            self._current_eval_id += 1
        except Exception:
            self._current_eval_id = getattr(self, "_current_eval_id", 1)
        # Record the logical evaluation time so switch throttling and
        # confirmation can be evaluated against the same simulated time used
        # by the plan builder (useful for deterministic tests which pass a
        # custom ``now_local`` into _build_plan).
        now_local = dt_util.now()
        self._current_eval_time = now_local
        domain_data = self.hass.data.get(DOMAIN, {})
        entries: Dict[str, Dict[str, Any]] = domain_data.get("entries", {})
        entry_data = entries.get(self.entry.entry_id, {})
        learning = entry_data.get("learning")

        try:
            raw_config = self._raw_config()
            # Use the configured coordinator-level default already stored in
            # self._confirmation_required; per-device overrides are applied
            # below when parsing devices. No global option is used.
            self._precharge_margin_on = self._option_float(
                CONF_PRECHARGE_MARGIN_ON, DEFAULT_PRECHARGE_MARGIN_ON
            )
            self._precharge_margin_off = self._option_float(
                CONF_PRECHARGE_MARGIN_OFF, DEFAULT_PRECHARGE_MARGIN_OFF
            )
            self._smart_start_margin = self._option_float(
                CONF_SMART_START_MARGIN, DEFAULT_SMART_START_MARGIN
            )
            self._default_learning_recent_sample_hours = max(
                0.25,
                self._option_float(
                    CONF_LEARNING_RECENT_SAMPLE_HOURS,
                    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
                ),
            )
            self._precharge_countdown_window = self._option_float(
                CONF_PRECHARGE_COUNTDOWN_WINDOW, DEFAULT_PRECHARGE_COUNTDOWN_WINDOW
            )

            for device in self._iter_device_configs(raw_config.get("devices") or []):
                device_window = device.learning_recent_sample_hours
                if device_window is None:
                    device_window = self._default_learning_recent_sample_hours
                device_window = max(0.25, min(48.0, float(device_window)))

                if learning is not None and hasattr(
                    learning, "set_recent_sample_window"
                ):
                    try:
                        learning.set_recent_sample_window(device_window)
                    except Exception as err:
                        _LOGGER.debug(
                            "Unable to update learning window for %s: %s",
                            device.name,
                            err,
                        )

                # configure per-device switch throttle from device options
                try:
                    ent = device.charger_switch
                    if device.switch_throttle_seconds is not None:
                        self._device_switch_throttle[ent] = max(
                            1.0, float(device.switch_throttle_seconds)
                        )
                    else:
                        self._device_switch_throttle[ent] = self._default_switch_throttle_seconds
                    # confirmation count
                    if device.switch_confirmation_count is not None:
                        self._desired_state_history.setdefault(ent, (False, 0))
                        # store per-entity confirmation requirement
                        self._device_switch_throttle.setdefault(f"{ent}::confirm", float(device.switch_confirmation_count))
                    else:
                        self._device_switch_throttle.setdefault(f"{ent}::confirm", float(self._confirmation_required))
                except Exception:
                    pass

                plan = await self._build_plan(
                    device,
                    now_local,
                    learning,
                    device_window,
                )
                if plan:
                    results[device.name] = plan.as_dict()

            if self._precharge_release:
                active = set(results.keys())
                for name in list(self._precharge_release.keys()):
                    if name not in active:
                        self._precharge_release.pop(name, None)
                        self._precharge_release_ready.pop(name, None)
            if self._drain_rate_cache:
                active = set(results.keys())
                for name in list(self._drain_rate_cache.keys()):
                    if name not in active:
                        self._drain_rate_cache.pop(name, None)

            self._state = results
            self._last_successful_update = dt_util.utcnow()
            return results

        except Exception:
            _LOGGER.exception("Smart Charger coordinator update failed")
            return self._state or {}

    async def _async_update_interval(self) -> None:
        """Adjust the update interval dynamically based on the battery state."""
        try:
            if not self._state:
                self.update_interval = timedelta(seconds=60)
                return

            avg_batt = sum(v.get("battery", 50) for v in self._state.values()) / len(
                self._state
            )
            if avg_batt < 30:
                new_interval = 120
            elif avg_batt < 60:
                new_interval = 60
            else:
                new_interval = 30

            current_interval = (
                self.update_interval.total_seconds() if self.update_interval else 60
            )
            if current_interval != new_interval:
                _LOGGER.debug(
                    "Adaptive update interval changed: %.0fs -> %.0fs (avg_batt=%.1f)",
                    current_interval,
                    new_interval,
                    avg_batt,
                )
                self.update_interval = timedelta(seconds=new_interval)

        except Exception as err:
            _LOGGER.debug("Adaptive interval calculation failed: %s", err)

    async def async_throttled_refresh(self) -> None:
        """Request an update only when the interval threshold is respected."""
        try:
            min_interval = (
                self.update_interval.total_seconds()
                if self.update_interval
                else UPDATE_INTERVAL
            )
        except Exception:
            min_interval = UPDATE_INTERVAL

        min_interval = max(30.0, float(min_interval))

        if self._last_successful_update:
            elapsed = (dt_util.utcnow() - self._last_successful_update).total_seconds()
            if elapsed < min_interval:
                _LOGGER.debug(
                    "Skipping refresh (elapsed=%.1fs, required=%.1fs)",
                    elapsed,
                    min_interval,
                )
                return

        await self.async_request_refresh()

    def _float_state(self, entity_id: Optional[str]) -> Optional[float]:
        if not entity_id:
            return None
        state_obj = self.hass.states.get(entity_id)
        if not state_obj or state_obj.state in UNKNOWN_STATES:
            return None
        try:
            return float(state_obj.state)
        except (TypeError, ValueError):
            _LOGGER.debug("Cannot convert state %s to float", state_obj.state)
            return None

    def _text_state(self, entity_id: Optional[str]) -> Optional[str]:
        if not entity_id:
            return None
        state_obj = self.hass.states.get(entity_id)
        if not state_obj or state_obj.state in UNKNOWN_STATES:
            return None
        return str(state_obj.state)

    def _charging_state(self, entity_id: Optional[str]) -> str:
        state = self._text_state(entity_id)
        if not state:
            return "unknown"
        lowered = state.lower()
        if lowered in CHARGING_STATES:
            return "charging"
        if lowered in FULL_STATES:
            return "full"
        if lowered in DISCHARGING_STATES:
            return "discharging"
        return "idle"

    def _resolve_alarm(self, device: DeviceConfig, now_local: datetime) -> datetime:
        """Find the next active alarm, respecting weekday specific entities."""

        today = now_local.date()
        weekday = now_local.weekday()

        for day_offset in range(0, 7):
            alarm_date = today + timedelta(days=day_offset)
            alarm_weekday = (weekday + day_offset) % 7
            alarm_entity = device.alarm_entity_for_today(alarm_weekday)
            alarm_time = self._parse_alarm_time(self._text_state(alarm_entity))

            alarm_dt = dt_util.as_local(datetime.combine(alarm_date, alarm_time))

            if day_offset == 0 and alarm_dt <= now_local:
                continue

            if alarm_dt > now_local:
                return alarm_dt

        """Fallback: default to an alarm tomorrow morning."""
        fallback_time = self._parse_alarm_time(None)
        return dt_util.as_local(
            datetime.combine(today + timedelta(days=1), fallback_time)
        )

    @staticmethod
    def _parse_alarm_time(value: Optional[str]) -> time:
        if not value:
            return time(hour=6, minute=30)

        stripped = value.strip()

        try:
            parsed = datetime.fromisoformat(stripped)
            return parsed.time()
        except ValueError:
            pass

        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(stripped, fmt).time()
            except ValueError:
                continue
        return time(hour=6, minute=30)

    def _avg_speed(
        self,
        device: DeviceConfig,
        learning,
        fallback: float = LEARNING_DEFAULT_SPEED,
    ) -> tuple[float, bool]:
        fallback = max(LEARNING_MIN_SPEED, min(LEARNING_MAX_SPEED, fallback))
        if device.use_predictive_mode and learning and hasattr(learning, "avg_speed"):
            try:
                speed = learning.avg_speed(device.name)
                if speed and float(speed) > 0:
                    return (
                        max(
                            LEARNING_MIN_SPEED, min(LEARNING_MAX_SPEED, float(speed))
                        ),
                        True,
                    )
            except Exception:
                _LOGGER.debug("Predictive avg_speed failed for %s", device.name)

            try:
                global_speed = learning.avg_speed()
                if global_speed and float(global_speed) > 0:
                    return (
                        max(
                            LEARNING_MIN_SPEED,
                            min(LEARNING_MAX_SPEED, float(global_speed)),
                        ),
                        False,
                    )
            except Exception:
                _LOGGER.debug("Predictive global avg_speed failed for %s", device.name)

        manual_state = self._float_state(device.avg_speed_sensor)
        if manual_state and manual_state > 0:
            return (
                max(LEARNING_MIN_SPEED, min(LEARNING_MAX_SPEED, manual_state)),
                True,
            )

        return fallback, False

    def _presence(self, device: DeviceConfig) -> tuple[bool, str]:
        state = self._text_state(device.presence_sensor)
        if state is None:
            return True, "unknown"
        lowered = state.lower()
        is_home = lowered in ("home", "on", "present", "true")
        return is_home, state

    def _predict_drain_rate(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        is_home: bool,
        charging_active: bool,
        learning_window_hours: float,
    ) -> tuple[float, float, list[str]]:
        base_reasons: list[str] = []
        observed_rate, stale_history = self._estimate_observed_drain(
            device,
            now_local=now_local,
            battery=battery,
            charging_active=charging_active,
            recent_window_hours=learning_window_hours,
        )
        if stale_history:
            base_reasons.append("stale_history")
        if charging_active and self._battery_history.get(device.name):
            base_reasons.append("charging_skip")

        rate = self._baseline_drain_rate(
            device,
            now_local=now_local,
            battery=battery,
            is_home=is_home,
            base_reasons=base_reasons,
        )

        if not base_reasons:
            base_reasons.append("fallback")

        rate, observed_flag = self._apply_observed_adjustment(rate, observed_rate)
        if observed_flag:
            base_reasons.append("observed_drain")

        rate, smoothed_flag = self._smooth_drain_rate(
            device,
            rate,
            observed=observed_flag,
            charging_active=charging_active,
            stale_history=stale_history,
        )
        base_reasons.append("ema_smoothing" if smoothed_flag else "seeded")

        confidence = self._drain_confidence(
            device,
            observed_rate=observed_rate,
            is_home=is_home,
            battery=battery,
            prior_exists=smoothed_flag,
        )

        clamped_rate = max(0.0, min(MAX_OBSERVED_DRAIN_RATE, rate))
        self._drain_rate_cache[device.name] = clamped_rate
        self._battery_history[device.name] = (now_local, battery, charging_active)

        return clamped_rate, confidence, base_reasons

    def _estimate_observed_drain(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        charging_active: bool,
        recent_window_hours: float,
    ) -> tuple[Optional[float], bool]:
        last_sample = self._battery_history.get(device.name)
        if not last_sample:
            return None, False
        prev_charging = False
        if len(last_sample) == 3:
            prev_ts, prev_level, prev_charging = last_sample  # type: ignore[misc]
        else:
            prev_ts, prev_level = last_sample  # type: ignore[misc]
        elapsed_hours = (now_local - prev_ts).total_seconds() / 3600
        if elapsed_hours <= 0:
            return None, False
        if elapsed_hours > recent_window_hours:
            return None, True
        if charging_active or prev_charging:
            return None, False
        delta = prev_level - battery
        if delta <= 0:
            return 0.0, False
        rate = max(0.0, delta / elapsed_hours)
        if rate <= 0.02:
            return 0.0, False
        return rate, False

    def _baseline_drain_rate(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        is_home: bool,
        base_reasons: list[str],
    ) -> float:
        if now_local.hour < 6:
            rate = 0.2
            base_reasons.append("night")
        elif now_local.hour < 22:
            rate = 0.4
            base_reasons.append("day")
        else:
            rate = 0.3
            base_reasons.append("late_evening")

        if not is_home:
            rate += 0.2
            base_reasons.append("away")
        if battery < device.min_level:
            rate += 0.1
            base_reasons.append("below_min")
        if battery < 20:
            rate += 0.05
            base_reasons.append("critical_reserve")
        if not device.use_predictive_mode:
            rate *= 0.9
            base_reasons.append("manual_mode_scaling")
        return max(0.0, rate)

    @staticmethod
    def _apply_observed_adjustment(
        rate: float, observed_rate: Optional[float]
    ) -> tuple[float, bool]:
        if observed_rate is None:
            return rate, False
        clamped_observed = max(0.0, min(MAX_OBSERVED_DRAIN_RATE, observed_rate))
        if clamped_observed <= 0.05:
            return 0.0, True

        if rate <= 0.0:
            return clamped_observed, True

        if clamped_observed > rate:
            if clamped_observed > max(1.5, rate * 2):
                blend = 0.25
            else:
                blend = 0.5
        else:
            blend = 0.6

        adjusted = rate + (clamped_observed - rate) * blend
        return min(MAX_OBSERVED_DRAIN_RATE, adjusted), True

    def _smooth_drain_rate(
        self,
        device: DeviceConfig,
        rate: float,
        *,
        observed: bool,
        charging_active: bool,
        stale_history: bool,
    ) -> tuple[float, bool]:
        prior = self._drain_rate_cache.get(device.name)
        if prior is None:
            return min(MAX_OBSERVED_DRAIN_RATE, max(0.0, rate)), False
        if charging_active and not observed:
            weight = 0.3
        elif stale_history and not observed:
            weight = 0.4
        else:
            weight = 0.65 if observed else 0.5
        smoothed = prior + (rate - prior) * weight
        smoothed = max(0.0, min(MAX_OBSERVED_DRAIN_RATE, smoothed))
        if charging_active and not observed:
            smoothed = min(smoothed, rate + 0.2)
        return smoothed, True

    def _drain_confidence(
        self,
        device: DeviceConfig,
        *,
        observed_rate: Optional[float],
        is_home: bool,
        battery: float,
        prior_exists: bool,
    ) -> float:
        confidence = 0.65
        if not prior_exists:
            confidence -= 0.1
        if not device.use_predictive_mode:
            confidence -= 0.05
        if not is_home:
            confidence -= 0.1
        if battery < device.min_level:
            confidence -= 0.05
        if observed_rate is not None:
            confidence += 0.1
        return max(0.2, min(0.95, confidence))

    async def _build_plan(
        self,
        device: DeviceConfig,
        now_local: datetime,
        learning,
        learning_window_hours: float,
    ) -> Optional[SmartChargePlan]:
        # When this helper is invoked directly (tests often call it with a
        # simulated ``now_local``), ensure we advance the per-evaluation
        # id and record the evaluation time so confirmation and throttle
        # logic use the same logical timebase.
        try:
            self._current_eval_id += 1
        except Exception:
            self._current_eval_id = getattr(self, "_current_eval_id", 1)
        self._current_eval_time = now_local

        battery = self._float_state(device.battery_sensor)
        if battery is None:
            _LOGGER.debug("No valid battery value for %s", device.name)
            return None

        charging_state = self._charging_state(device.charging_sensor)
        charging_active = charging_state == "charging"
        avg_speed, speed_confident = self._avg_speed(device, learning)
        is_home, presence_state = self._presence(device)
        alarm_dt = self._resolve_alarm(device, now_local)

        hours_until_alarm = max(0.0, (alarm_dt - now_local).total_seconds() / 3600)

        drain_rate, drain_confidence, drain_basis = self._predict_drain_rate(
            device,
            now_local=now_local,
            battery=battery,
            is_home=is_home,
            charging_active=charging_active,
            learning_window_hours=learning_window_hours,
        )

        expected_drain = max(0.0, hours_until_alarm * drain_rate)
        predicted_level = max(0.0, battery - expected_drain)

        charge_deficit = max(0.0, device.target_level - predicted_level)
        if charge_deficit > 0.0:
            if avg_speed > 0.0:
                duration_hours = charge_deficit / max(avg_speed, 1e-3)
                duration_min = min(duration_hours * 60.0, 24 * 60)
                if not speed_confident:
                    heuristic_min = (
                        charge_deficit * DEFAULT_FALLBACK_MINUTES_PER_PERCENT
                    )
                    heuristic_min = min(heuristic_min, 24 * 60)
                    if heuristic_min > 0:
                        duration_min = min(duration_min, heuristic_min * 1.2)
            else:
                duration_min = 24 * 60
            if hours_until_alarm > 0:
                min_window_hours = 0.25
                duration_min = max(duration_min, min_window_hours * 60.0)
                duration_min = min(duration_min, hours_until_alarm * 60.0)
        else:
            duration_min = 0.0

        start_time, smart_start_active, duration_min = self._resolve_start_window(
            alarm_dt=alarm_dt,
            duration_min=duration_min,
            charge_deficit=charge_deficit,
        )
        main_duration_min = duration_min

        charger_state = self.hass.states.get(device.charger_switch)
        if charger_state and charger_state.state not in UNKNOWN_STATES:
            charger_available = True
            charger_state_value = str(charger_state.state).lower()
        else:
            charger_available = False
            charger_state_value = ""
        charger_is_on = charger_available and charger_state_value in (
            "on",
            "charging",
            STATE_ON,
        )
        (
            precharge_required,
            release_level,
            margin_on,
            margin_off,
            smart_margin,
            forecast_holdoff,
        ) = self._evaluate_precharge_state(
            device,
            now_local=now_local,
            battery=battery,
            predicted_level=predicted_level,
            expected_drain=expected_drain,
            is_home=is_home,
            smart_start_active=smart_start_active,
            start_time=start_time,
        )

        charger_expected_on = await self._apply_charger_logic(
            device,
            now_local=now_local,
            battery=battery,
            charger_is_on=bool(charger_is_on),
            charger_available=charger_available,
            is_home=is_home,
            start_time=start_time,
            smart_start_active=smart_start_active,
            precharge_required=precharge_required,
            release_level=release_level,
            margin_on=margin_on,
            smart_margin=smart_margin,
            charge_deficit=charge_deficit,
            predicted_level=predicted_level,
            forecast_holdoff=forecast_holdoff,
        )

        # Record the observed desired state so confirmation counters are
        # maintained across coordinator evaluations even when no switch
        # service was invoked during this run.
        try:
            self._record_desired_state(device.charger_switch, bool(charger_expected_on))
        except Exception:
            # Non-critical; avoid failing the whole plan generation on record
            pass

        precharge_active = (precharge_required and charger_expected_on) or (
            charger_expected_on
            and smart_start_active
            and start_time is not None
            and now_local < start_time
        )

        total_duration_min = main_duration_min
        precharge_duration_min: Optional[float] = None
        if release_level is not None:
            if avg_speed > 0:
                precharge_deficit = max(0.0, release_level - battery)
                if precharge_deficit > 0:
                    precharge_duration_min = min(
                        (precharge_deficit / max(avg_speed, 1e-3)) * 60.0,
                        24 * 60,
                    )
            elif precharge_required:
                precharge_duration_min = 24 * 60

        if precharge_duration_min is not None:
            total_duration_min = min(total_duration_min + precharge_duration_min, 48 * 60)

        if precharge_active and precharge_duration_min is not None:
            display_duration_min = precharge_duration_min
        else:
            display_duration_min = main_duration_min

        total_duration_min = max(0.0, total_duration_min)

        return SmartChargePlan(
            battery=battery,
            target=device.target_level,
            avg_speed=avg_speed,
            duration_min=display_duration_min,
            charge_duration_min=main_duration_min,
            total_duration_min=total_duration_min,
            precharge_duration_min=precharge_duration_min,
            alarm_time=alarm_dt,
            start_time=start_time,
            predicted_drain=expected_drain,
            predicted_level_at_alarm=predicted_level,
            drain_rate=drain_rate,
            drain_confidence=drain_confidence,
            drain_basis=tuple(drain_basis),
            smart_start_active=smart_start_active,
            precharge_level=device.precharge_level,
            precharge_margin_on=margin_on,
            precharge_margin_off=margin_off,
            smart_start_margin=smart_margin,
            precharge_active=precharge_active,
            precharge_release_level=release_level,
            charging_state=charging_state,
            presence_state=presence_state,
            last_update=dt_util.utcnow(),
        )

    @staticmethod
    def _resolve_start_window(
        *,
        alarm_dt: datetime,
        duration_min: float,
        charge_deficit: float,
    ) -> tuple[Optional[datetime], bool, float]:
        if charge_deficit <= 0.0 or duration_min <= 0.0:
            return None, False, 0.0
        start_time = alarm_dt - timedelta(minutes=duration_min)
        return start_time, True, duration_min

    def _evaluate_precharge_state(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        predicted_level: float,
        expected_drain: float,
        is_home: bool,
        smart_start_active: bool,
        start_time: Optional[datetime],
    ) -> tuple[bool, Optional[float], float, float, float, bool]:
        precharge_required = False
        release_level = self._precharge_release.get(device.name)
        previous_release = release_level
        release_ready_at = self._precharge_release_ready.get(device.name)
        margin_on = (
            device.precharge_margin_on
            if device.precharge_margin_on is not None
            else self._precharge_margin_on
        )
        margin_off = (
            device.precharge_margin_off
            if device.precharge_margin_off is not None
            else self._precharge_margin_off
        )
        smart_margin = (
            device.smart_start_margin
            if device.smart_start_margin is not None
            else self._smart_start_margin
        )
        forecast_holdoff = False

        if not is_home:
            if release_level is not None:
                self._precharge_release.pop(device.name, None)
                self._precharge_release_ready.pop(device.name, None)
            release_level = None
        elif device.precharge_level > device.min_level:
            trigger_threshold = device.precharge_level - margin_on
            predicted_threshold = device.precharge_level + margin_on
            near_precharge_window = (
                battery
                <= device.precharge_level + self._precharge_countdown_window
            )

            should_latch = False
            if battery <= trigger_threshold:
                should_latch = True
            elif predicted_level <= trigger_threshold and near_precharge_window:
                should_latch = True

            if should_latch:
                extra_margin = max(margin_off, expected_drain * 0.4)
                release_cap = device.target_level
                if smart_start_active and start_time and now_local < start_time:
                    release_cap = max(
                        device.precharge_level,
                        device.target_level - smart_margin,
                    )
                release_level = min(
                    release_cap,
                    device.precharge_level + extra_margin,
                )
                release_level = max(device.precharge_level, release_level)
                self._precharge_release[device.name] = release_level
                self._precharge_release_ready.pop(device.name, None)
                if previous_release != release_level:
                    self._log_action(
                        device.name,
                        logging.DEBUG,
                        "[Precharge] %s latched until %.1f%% (margins on %.2f/off %.2f)",
                        device.name,
                        release_level,
                        margin_on,
                        margin_off,
                    )
                precharge_required = True
            elif release_level is not None:
                previously_in_range = release_ready_at is not None
                in_range = (
                    release_level is not None
                    and battery >= release_level
                    and predicted_level >= predicted_threshold
                )
                near_precharge = near_precharge_window

                if not previously_in_range:
                    if in_range:
                        if near_precharge:
                            release_ready_at = now_local + PRECHARGE_RELEASE_HYSTERESIS
                            self._precharge_release_ready[device.name] = release_ready_at
                            self._log_action(
                                device.name,
                                logging.DEBUG,
                                "[Precharge] %s release countdown started; clears at %s",
                                device.name,
                                release_ready_at.isoformat(),
                            )
                            precharge_required = True
                        else:
                            self._precharge_release.pop(device.name, None)
                            self._precharge_release_ready.pop(device.name, None)
                            self._log_action(
                                device.name,
                                logging.DEBUG,
                                "[Precharge] %s release cleared (battery %.1f%%, predicted %.1f%%)",
                                device.name,
                                battery,
                                predicted_level,
                            )
                            release_level = None
                    else:
                        self._precharge_release_ready.pop(device.name, None)
                        precharge_required = True
                else:
                    if in_range and near_precharge:
                        if release_ready_at is not None and now_local >= release_ready_at:
                            self._precharge_release.pop(device.name, None)
                            self._precharge_release_ready.pop(device.name, None)
                            self._log_action(
                                device.name,
                                logging.DEBUG,
                                "[Precharge] %s release window cleared",
                                device.name,
                            )
                            release_level = None
                        else:
                            precharge_required = True
                    elif not near_precharge:
                        self._precharge_release.pop(device.name, None)
                        self._precharge_release_ready.pop(device.name, None)
                        self._log_action(
                            device.name,
                            logging.DEBUG,
                            "[Precharge] %s release cleared immediately (battery %.1f%%)",
                            device.name,
                            battery,
                        )
                        release_level = None
                    else:
                        # Keep countdown running even if the level briefly dips again.
                        precharge_required = True

        if release_level is not None and not precharge_required and is_home:
            if battery < release_level or predicted_level < device.precharge_level:
                precharge_required = True
                self._log_action(
                    device.name,
                    logging.DEBUG,
                    "[Precharge] %s staying active until %.1f%% release threshold",
                    device.name,
                    release_level,
                )

        if (
            device.use_predictive_mode
            and not precharge_required
            and release_level is None
            and battery > device.precharge_level + self._precharge_countdown_window
            and predicted_level <= device.precharge_level - margin_on
        ):
            forecast_holdoff = True

        return (
            precharge_required,
            release_level,
            margin_on,
            margin_off,
            smart_margin,
            forecast_holdoff,
        )

    async def _async_switch_call(self, action: str, service_data: Dict[str, Any]) -> bool:
        entity_id = service_data.get("entity_id")
        if not self.hass.services.has_service("switch", action):
            _LOGGER.debug(
                "Service switch.%s unavailable; skipping request for %s",
                action,
                entity_id,
            )
            return False
        await self.hass.services.async_call(
            "switch", action, service_data, blocking=False
        )
        if entity_id:
            # Prefer the coordinator's logical evaluation time when set so
            # tests that pass a simulated now_local compare consistently.
            self._last_switch_time[entity_id] = getattr(
                self, "_current_eval_time", dt_util.utcnow()
            )
        return True

    def _record_desired_state(self, entity_id: str, desired: bool) -> None:
        """Record an observed desired state for confirmation counting.

        This increments the consecutive observation counter for the given
        entity, or resets it to 1 when the desired state changes.
        """
        if not entity_id:
            return
        # Avoid recording more than once per coordinator evaluation.
        last = self._last_recorded_eval.get(entity_id)
        if last == self._current_eval_id:
            return
        self._last_recorded_eval[entity_id] = self._current_eval_id

        confirm_key = f"{entity_id}::confirm"
        required = int(
            self._device_switch_throttle.get(confirm_key, float(self._confirmation_required))
        )
        hist = self._desired_state_history.get(entity_id, (desired, 0))
        if hist[0] == desired:
            count = min(required, hist[1] + 1)
        else:
            count = 1
        self._desired_state_history[entity_id] = (desired, count)

    async def _maybe_switch(self, action: str, service_data: Dict[str, Any], desired: bool, force: bool = False) -> bool:
        """Issue a switch call after throttle and confirmation debounce.

        If force is True, bypass confirmation and throttle (for emergency actions).
        """
        entity_id = service_data.get("entity_id")
        if not entity_id:
            return False

        # Use the coordinator's current evaluation time when available so
        # tests that simulate future times behave deterministically. Fall
        # back to real time if not set.
        now = getattr(self, "_current_eval_time", None) or dt_util.utcnow()

        # Throttle check (per-device configured)
        if not force:
            throttle = self._device_switch_throttle.get(
                entity_id, self._default_switch_throttle_seconds
            )
            last = self._last_switch_time.get(entity_id)

            # Confirmation debounce: per-device override available. Record the
            # observed desired state for confirmation counting. The helper will
            # increment or reset the consecutive counter as appropriate.
            self._record_desired_state(entity_id, desired)
            confirm_key = f"{entity_id}::confirm"
            required = int(
                self._device_switch_throttle.get(confirm_key, float(self._confirmation_required))
            )
            hist = self._desired_state_history.get(entity_id, (desired, 0))
            count = hist[1]

            # If we haven't yet observed the required number of consecutive
            # confirmations, wait (regardless of throttle state).
            if count < required:
                _LOGGER.debug(
                    "Waiting for confirmation for %s -> desired=%s (count=%d/%d)",
                    entity_id,
                    desired,
                    count,
                    required,
                )
                return False

            # Throttle check (per-device configured): even if we've reached the
            # confirmation count, do not issue switches while inside the throttle
            # window. The confirmation counter will continue to advance in the
            # background and the next coordinator evaluation after the throttle
            # expires will proceed to call the switch.
            if last and (now - last).total_seconds() < float(throttle):
                _LOGGER.debug(
                    "Throttling switch.%s for %s (last %.1fs ago, throttle=%.1fs)",
                    action,
                    entity_id,
                    (now - last).total_seconds(),
                    float(throttle),
                )
                return False

        # Clear history and call switch
        self._desired_state_history.pop(entity_id, None)
        result = await self._async_switch_call(action, service_data)
        if result:
            self._last_switch_time[entity_id] = getattr(
                self, "_current_eval_time", dt_util.utcnow()
            )
        return result

    async def _apply_charger_logic(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        charger_is_on: bool,
        charger_available: bool,
        is_home: bool,
        start_time: Optional[datetime],
        smart_start_active: bool,
        precharge_required: bool,
        release_level: Optional[float],
        margin_on: float,
        smart_margin: float,
        charge_deficit: float,
        predicted_level: float,
        forecast_holdoff: bool,
    ) -> bool:
        charger_ent = device.charger_switch
        expected_on = charger_is_on

        service_data = {"entity_id": charger_ent}
        device_name = device.name
        window_imminent = (
            smart_start_active
            and start_time is not None
            and start_time > now_local
            and (start_time - now_local) <= timedelta(seconds=5)
        )

        if battery <= device.min_level and not charger_is_on:
            self._log_action(
                device_name,
                logging.WARNING,
                "[EmergencyCharge] %s below minimum level %.1f%% -> starting charging immediately (%s)",
                device_name,
                device.min_level,
                charger_ent,
            )
            await self._maybe_switch("turn_on", service_data, desired=True, force=True)
            return True

        if not charger_available:
            if precharge_required or smart_start_active:
                self._log_action(
                    device_name,
                    logging.DEBUG,
                    "[Charger] %s unavailable -> skipping control actions",
                    device_name,
                )
            return expected_on

        if (
            forecast_holdoff
            and smart_start_active
            and not precharge_required
        ):
            window_threshold = device.precharge_level + self._precharge_countdown_window
            if charger_is_on:
                self._log_action(
                    device_name,
                    logging.INFO,
                    "[SmartStart] %s deferring predictive start until within %.1f%% window -> pausing charger (%s)",
                    device_name,
                    window_threshold,
                    charger_ent,
                )
                await self._maybe_switch("turn_off", service_data, desired=False)
                return False

            if start_time and now_local >= start_time:
                self._log_action(
                    device_name,
                    logging.DEBUG,
                    "[SmartStart] %s ignoring distant drain forecast (battery %.1f%%, guard %.1f%%)",
                    device_name,
                    battery,
                    window_threshold,
                )
                return expected_on

        if (
            smart_start_active
            and start_time
            and now_local >= start_time
            and not charger_is_on
            and not precharge_required
            and not forecast_holdoff
        ):
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStart] Charging start time reached for %s -> activating charger (%s)",
                device_name,
                charger_ent,
            )
            await self._maybe_switch("turn_on", service_data, desired=True)
            return True

        if charger_is_on and battery >= device.target_level:
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStop] %s reached target level %.1f%% -> deactivating charger (%s)",
                device_name,
                battery,
                charger_ent,
            )
            await self._maybe_switch("turn_off", service_data, desired=False)
            return False

        if (
            charger_is_on
            and smart_start_active
            and start_time
            and now_local < start_time
            and not precharge_required
            and charge_deficit <= 0
            and battery >= device.target_level - smart_margin
            and not window_imminent
        ):
            self._log_action(
                device_name,
                logging.INFO,
                "[Precharge] %s reached the safety level and waits for the start at %s -> pausing charger (%s)",
                device_name,
                start_time.isoformat(),
                charger_ent,
            )
            await self._maybe_switch("turn_off", service_data, desired=False)
            return False

        if (
            charger_is_on
            and smart_start_active
            and start_time
            and now_local < start_time
            and not precharge_required
            and charge_deficit <= 0
            and predicted_level >= device.target_level
            and not window_imminent
        ):
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStart] %s scheduled for %s -> pausing charger until window opens (%s)",
                device_name,
                start_time.isoformat(),
                charger_ent,
            )
            await self._maybe_switch("turn_off", service_data, desired=False)
            return False

        if (
            charger_is_on
            and smart_start_active
            and start_time
            and now_local <= start_time
            and not precharge_required
            and not window_imminent
        ):
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStart] %s finished precharge and will wait for the window at %s -> pausing charger (%s)",
                device_name,
                start_time.isoformat(),
                charger_ent,
            )
            # Use _maybe_switch so throttle and confirmation debounce apply
            # (previously used _async_switch_call here which bypassed throttling
            # and could cause repeated immediate turn_off calls during precharge
            # pause conditions).
            await self._maybe_switch("turn_off", service_data, desired=False)
            return False

        if (
            precharge_required
            and release_level is None
            and battery >= device.precharge_level + margin_on
            and predicted_level >= device.precharge_level + margin_on
        ):
            self._log_action(
                device_name,
                logging.DEBUG,
                "[Precharge] %s skipping precharge (battery %.1f%%, predicted %.1f%%, threshold %.1f%%)",
                device_name,
                battery,
                predicted_level,
                device.precharge_level + margin_on,
            )
            precharge_required = False

        if precharge_required:
            target_release = (
                release_level if release_level is not None else device.precharge_level
            )
            start_threshold = max(target_release - margin_on, device.precharge_level)

            if (
                not expected_on
                and battery >= start_threshold
                and not window_imminent
                and predicted_level >= device.precharge_level
            ):
                self._log_action(
                    device_name,
                    logging.DEBUG,
                    "[Precharge] %s already above %.1f%% -> waiting to latch charger",
                    device_name,
                    target_release,
                )
                return expected_on

            if expected_on and battery >= target_release and not window_imminent:
                if predicted_level < device.precharge_level:
                    self._log_action(
                        device_name,
                        logging.DEBUG,
                        "[Precharge] %s waiting for forecast to recover (battery %.1f%% -> predicted %.1f%%)",
                        device_name,
                        battery,
                        predicted_level,
                    )
                    return True

                self._log_action(
                    device_name,
                    logging.INFO,
                    "[Precharge] %s reached release %.1f%% -> pausing charger (%s)",
                    device_name,
                    target_release,
                    charger_ent,
                )
                await self._maybe_switch("turn_off", service_data, desired=False)
                return False

            if not expected_on:
                self._log_action(
                    device_name,
                    logging.INFO,
                    "[Precharge] %s requires precharge -> activating charger %s until %.1f%%",
                    device_name,
                    charger_ent,
                    target_release,
                )
                await self._maybe_switch("turn_on", service_data, desired=True)
                return True

            self._log_action(
                device_name,
                logging.DEBUG,
                "[Precharge] Keeping charger on for %s until %.1f%%",
                device_name,
                target_release,
            )
            return True

        return expected_on

    # NOTE: unreachable - kept for clarity

    def _log_action(
        self, device_name: str, level: int, message: str, *args: Any
    ) -> None:
        rendered = message % args if args else message
        if self._last_action_log.get(device_name) == rendered:
            _LOGGER.debug(
                "Skipping duplicate action log for %s: %s", device_name, rendered
            )
            return
        self._last_action_log[device_name] = rendered
        _LOGGER.log(level, rendered)
