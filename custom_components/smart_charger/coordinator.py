from __future__ import annotations

import logging
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
    CONF_PRESENCE_SENSOR,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    CONF_SMART_START_MARGIN,
    DEFAULT_TARGET_LEVEL,
    DEFAULT_PRECHARGE_MARGIN_OFF,
    DEFAULT_PRECHARGE_MARGIN_ON,
    DEFAULT_SMART_START_MARGIN,
    DEFAULT_PRECHARGE_COUNTDOWN_WINDOW,
    DISCHARGING_STATES,
    DOMAIN,
    LEARNING_DEFAULT_SPEED,
    LEARNING_MAX_SPEED,
    LEARNING_MIN_SPEED,
    FULL_STATES,
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
    charging_sensor: Optional[str] = None
    avg_speed_sensor: Optional[str] = None
    presence_sensor: Optional[str] = None
    alarm_mode: str = ALARM_MODE_SINGLE
    alarm_entity: Optional[str] = None
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
    charging_state: str
    presence_state: str
    last_update: datetime

    def as_dict(self) -> Dict[str, Any]:
        return {
            "battery": round(self.battery, 1),
            "target": round(self.target, 1),
            "avg_speed": round(self.avg_speed, 3),
            "duration_min": round(self.duration_min, 1),
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
        self._battery_history: Dict[str, tuple[datetime, float]] = {}
        self._precharge_margin_on = DEFAULT_PRECHARGE_MARGIN_ON
        self._precharge_margin_off = DEFAULT_PRECHARGE_MARGIN_OFF
        self._smart_start_margin = DEFAULT_SMART_START_MARGIN
        self._precharge_countdown_window = DEFAULT_PRECHARGE_COUNTDOWN_WINDOW

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
        domain_data = self.hass.data.get(DOMAIN, {})
        entries: Dict[str, Dict[str, Any]] = domain_data.get("entries", {})
        entry_data = entries.get(self.entry.entry_id, {})
        learning = entry_data.get("learning")
        now_local = dt_util.now()

        try:
            raw_config = self._raw_config()
            self._precharge_margin_on = self._option_float(
                CONF_PRECHARGE_MARGIN_ON, DEFAULT_PRECHARGE_MARGIN_ON
            )
            self._precharge_margin_off = self._option_float(
                CONF_PRECHARGE_MARGIN_OFF, DEFAULT_PRECHARGE_MARGIN_OFF
            )
            self._smart_start_margin = self._option_float(
                CONF_SMART_START_MARGIN, DEFAULT_SMART_START_MARGIN
            )
            self._precharge_countdown_window = self._option_float(
                CONF_PRECHARGE_COUNTDOWN_WINDOW, DEFAULT_PRECHARGE_COUNTDOWN_WINDOW
            )

            for device in self._iter_device_configs(raw_config.get("devices") or []):
                plan = await self._build_plan(device, now_local, learning)
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
    ) -> tuple[float, float, list[str]]:
        base_reasons: list[str] = []
        observed_rate = self._estimate_observed_drain(
            device, now_local=now_local, battery=battery, charging_active=charging_active
        )
        if charging_active and self._battery_history.get(device.name):
            base_reasons.append("charging_skip")

        rate = self._baseline_drain_rate(
            device,
            now_local=now_local,
            battery=battery,
            is_home=is_home,
            base_reasons=base_reasons,
        )

        rate, observed_flag = self._apply_observed_adjustment(rate, observed_rate)
        if observed_flag:
            base_reasons.append("observed_drain")

        rate, smoothed_flag = self._smooth_drain_rate(device, rate)
        base_reasons.append("ema_smoothing" if smoothed_flag else "seeded")

        confidence = self._drain_confidence(
            device,
            observed_rate=observed_rate,
            is_home=is_home,
            battery=battery,
            prior_exists=smoothed_flag,
        )

        self._drain_rate_cache[device.name] = rate
        self._battery_history[device.name] = (now_local, battery)

        return rate, confidence, base_reasons

    def _estimate_observed_drain(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        charging_active: bool,
    ) -> Optional[float]:
        last_sample = self._battery_history.get(device.name)
        if not last_sample:
            return None
        if charging_active:
            return None
        prev_ts, prev_level = last_sample
        elapsed_hours = (now_local - prev_ts).total_seconds() / 3600
        if elapsed_hours <= 0:
            return None
        delta = prev_level - battery
        if delta <= 0.2:
            return None
        return max(0.0, delta / elapsed_hours)

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
        return rate + (observed_rate - rate) * 0.3, True

    def _smooth_drain_rate(
        self, device: DeviceConfig, rate: float
    ) -> tuple[float, bool]:
        prior = self._drain_rate_cache.get(device.name)
        if prior is None:
            return rate, False
        smoothed = prior + (rate - prior) * 0.5
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
    ) -> Optional[SmartChargePlan]:
        battery = self._float_state(device.battery_sensor)
        if battery is None:
            _LOGGER.debug("No valid battery value for %s", device.name)
            return None

        charging_state = self._charging_state(device.charging_sensor)
        avg_speed, speed_confident = self._avg_speed(device, learning)
        is_home, presence_state = self._presence(device)
        alarm_dt = self._resolve_alarm(device, now_local)

        hours_until_alarm = max(0.0, (alarm_dt - now_local).total_seconds() / 3600)

        drain_rate, drain_confidence, drain_basis = self._predict_drain_rate(
            device,
            now_local=now_local,
            battery=battery,
            is_home=is_home,
            charging_active=(charging_state == "charging"),
        )

        expected_drain = max(0.0, hours_until_alarm * drain_rate)
        predicted_level = max(0.0, battery - expected_drain)

        charge_deficit = max(0.0, device.target_level - predicted_level)
        if charge_deficit > 0.0:
            if speed_confident and avg_speed > 0.0:
                duration_min = min(charge_deficit / avg_speed, 24 * 60)
            else:
                # Without a usable speed we fall back to the longest window to keep charging active.
                duration_min = 24 * 60
        else:
            duration_min = 0.0

        start_time, smart_start_active, duration_min = self._resolve_start_window(
            alarm_dt=alarm_dt,
            duration_min=duration_min,
            charge_deficit=charge_deficit,
        )

        charger_state = self.hass.states.get(device.charger_switch)
        charger_is_on = charger_state and str(charger_state.state).lower() in (
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
            is_home=is_home,
            start_time=start_time,
            smart_start_active=smart_start_active,
            precharge_required=precharge_required,
            release_level=release_level,
            smart_margin=smart_margin,
            charge_deficit=charge_deficit,
            predicted_level=predicted_level,
        )

        precharge_active = (precharge_required and charger_expected_on) or (
            charger_expected_on
            and smart_start_active
            and start_time is not None
            and now_local < start_time
        )

        return SmartChargePlan(
            battery=battery,
            target=device.target_level,
            avg_speed=avg_speed,
            duration_min=duration_min,
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
    ) -> tuple[bool, Optional[float], float, float, float]:
        precharge_required = False
        release_level = self._precharge_release.get(device.name)
        previous_release = release_level
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

        if not is_home:
            if release_level is not None:
                self._precharge_release.pop(device.name, None)
                self._precharge_release_ready.pop(device.name, None)
            release_level = None
        elif device.precharge_level > device.min_level:
            trigger_threshold = device.precharge_level - margin_on
            predicted_threshold = device.precharge_level + margin_on

            if battery <= trigger_threshold or predicted_level <= trigger_threshold:
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
                if battery >= release_level and predicted_level >= predicted_threshold:
                    near_precharge = (
                        battery
                        <= device.precharge_level + self._precharge_countdown_window
                    )
                    ready_at = self._precharge_release_ready.get(device.name)
                    if not near_precharge:
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
                    elif ready_at is None:
                        ready_at = now_local + PRECHARGE_RELEASE_HYSTERESIS
                        self._precharge_release_ready[device.name] = ready_at
                        self._log_action(
                            device.name,
                            logging.DEBUG,
                            "[Precharge] %s release countdown started; clears at %s",
                            device.name,
                            ready_at.isoformat(),
                        )
                        precharge_required = True
                    elif now_local >= ready_at:
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
                else:
                    self._precharge_release_ready.pop(device.name, None)
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

        return precharge_required, release_level, margin_on, margin_off, smart_margin

    async def _apply_charger_logic(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        charger_is_on: bool,
        is_home: bool,
        start_time: Optional[datetime],
        smart_start_active: bool,
        precharge_required: bool,
        release_level: Optional[float],
        smart_margin: float,
        charge_deficit: float,
        predicted_level: float,
    ) -> bool:
        charger_ent = device.charger_switch
        expected_on = charger_is_on

        service_data = {"entity_id": charger_ent}
        device_name = device.name
        window_imminent = (
            smart_start_active
            and start_time is not None
            and now_local + timedelta(seconds=5) >= start_time
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
            await self.hass.services.async_call(
                "switch", "turn_on", service_data, blocking=False
            )
            return True

        if (
            smart_start_active
            and start_time
            and now_local >= start_time
            and not charger_is_on
            and not precharge_required
        ):
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStart] Charging start time reached for %s -> activating charger (%s)",
                device_name,
                charger_ent,
            )
            await self.hass.services.async_call(
                "switch", "turn_on", service_data, blocking=False
            )
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
            await self.hass.services.async_call(
                "switch", "turn_off", service_data, blocking=False
            )
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
            await self.hass.services.async_call(
                "switch", "turn_off", service_data, blocking=False
            )
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
            await self.hass.services.async_call(
                "switch", "turn_off", service_data, blocking=False
            )
            return False

        if precharge_required:
            target_release = (
                release_level if release_level is not None else device.precharge_level
            )
            if not expected_on:
                self._log_action(
                    device_name,
                    logging.INFO,
                    "[Precharge] %s requires precharge -> activating charger %s until %.1f%%",
                    device_name,
                    charger_ent,
                    target_release,
                )
                await self.hass.services.async_call(
                    "switch", "turn_on", service_data, blocking=False
                )
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
