from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Any, Dict, Iterable, Mapping, Optional
import logging

from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    UPDATE_INTERVAL,
    CONF_BATTERY_SENSOR,
    CONF_CHARGING_SENSOR,
    CONF_ALARM_ENTITY,
    CONF_AVG_SPEED_SENSOR,
    CONF_PRESENCE_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_TARGET_LEVEL,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    DEFAULT_TARGET_LEVEL,
    CONF_ALARM_MODE,
    ALARM_MODE_SINGLE,
    ALARM_MODE_PER_DAY,
    CONF_ALARM_MONDAY,
    CONF_ALARM_TUESDAY,
    CONF_ALARM_WEDNESDAY,
    CONF_ALARM_THURSDAY,
    CONF_ALARM_FRIDAY,
    CONF_ALARM_SATURDAY,
    CONF_ALARM_SUNDAY,
    UNKNOWN_STATES,
    CHARGING_STATES,
    DISCHARGING_STATES,
    FULL_STATES,
)

_LOGGER = logging.getLogger(__name__)


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
    smart_start_active: bool
    precharge_level: float
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
            "start_time": dt_util.as_local(self.start_time).isoformat() if self.start_time else None,
            "predicted_drain": round(self.predicted_drain, 2),
            "predicted_level_at_alarm": round(self.predicted_level_at_alarm, 1),
            "smart_start_active": self.smart_start_active,
            "precharge_level": round(self.precharge_level, 1),
            "precharge_active": self.precharge_active,
            "charging_state": self.charging_state,
            "presence_state": self.presence_state,
            "last_update": dt_util.now().isoformat(),
        }


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

    @property
    def profiles(self) -> Dict[str, Dict[str, Any]]:
        return self._state or {}

    def _raw_config(self) -> Dict[str, Any]:
        self.config = {**dict(self.entry.data), **getattr(self.entry, "options", {})}
        return self.config

    def _iter_device_configs(self) -> Iterable[DeviceConfig]:
        devices = self._raw_config().get("devices") or []
        for raw in devices:
            try:
                yield DeviceConfig.from_dict(raw)
            except Exception as err:
                _LOGGER.warning("Skipping invalid device configuration %s: %s", raw, err)

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        domain_data = self.hass.data.get(DOMAIN, {})
        entries: Dict[str, Dict[str, Any]] = domain_data.get("entries", {})
        entry_data = entries.get(self.entry.entry_id, {})
        learning = entry_data.get("learning")
        now_local = dt_util.now()

        try:
            for device in self._iter_device_configs():
                plan = await self._build_plan(device, now_local, learning)
                if plan:
                    results[device.name] = plan.as_dict()

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

            avg_batt = sum(v.get("battery", 50) for v in self._state.values()) / len(self._state)
            if avg_batt < 30:
                new_interval = 120
            elif avg_batt < 60:
                new_interval = 60
            else:
                new_interval = 30

            current_interval = self.update_interval.total_seconds() if self.update_interval else 60
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
            min_interval = self.update_interval.total_seconds() if self.update_interval else UPDATE_INTERVAL
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
        return dt_util.as_local(datetime.combine(today + timedelta(days=1), fallback_time))

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

    def _avg_speed(self, device: DeviceConfig, learning, fallback: float = 1.0) -> float:
        if device.use_predictive_mode and learning and hasattr(learning, "avg_speed"):
            try:
                speed = learning.avg_speed(device.name)
                if speed:
                    return max(0.1, float(speed))
            except Exception:
                _LOGGER.debug("Predictive avg_speed failed for %s", device.name)

        manual_state = self._float_state(device.avg_speed_sensor)
        if manual_state and manual_state > 0:
            return manual_state

        return fallback

    def _presence(self, device: DeviceConfig) -> tuple[bool, str]:
        state = self._text_state(device.presence_sensor)
        if state is None:
            return True, "unknown"
        lowered = state.lower()
        is_home = lowered in ("home", "on", "present", "true")
        return is_home, state

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
        avg_speed = max(0.1, self._avg_speed(device, learning))
        is_home, presence_state = self._presence(device)
        alarm_dt = self._resolve_alarm(device, now_local)

        diff = max(0.0, device.target_level - battery)
        duration_min = min(diff / avg_speed, 24 * 60)
        hours_until_alarm = max(0.0, (alarm_dt - now_local).total_seconds() / 3600)

        base_drain = 0.2 if now_local.hour < 6 else (0.4 if now_local.hour < 22 else 0.3)
        if not is_home:
            base_drain += 0.2
        if battery < 20:
            base_drain += 0.1
        if not device.use_predictive_mode:
            base_drain *= 0.9

        expected_drain = hours_until_alarm * base_drain
        predicted_level = max(0.0, battery - expected_drain)

        if predicted_level < device.target_level:
            start_time = alarm_dt - timedelta(minutes=duration_min)
            smart_start_active = True
        else:
            start_time = None
            smart_start_active = False
            duration_min = 0.0

        charger_state = self.hass.states.get(device.charger_switch)
        charger_is_on = charger_state and str(charger_state.state).lower() in ("on", "charging", STATE_ON)

        precharge_margin = 0.5
        precharge_required = (
            is_home
            and device.precharge_level > device.min_level
            and (
                battery <= device.precharge_level - precharge_margin
                or predicted_level <= device.precharge_level - precharge_margin
            )
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
        )

        precharge_active = (
            (precharge_required and charger_expected_on)
            or (
                charger_expected_on
                and smart_start_active
                and start_time is not None
                and now_local < start_time
            )
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
            smart_start_active=smart_start_active,
            precharge_level=device.precharge_level,
            precharge_active=precharge_active,
            charging_state=charging_state,
            presence_state=presence_state,
            last_update=dt_util.utcnow(),
        )

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
    ) -> bool:
        charger_ent = device.charger_switch
        expected_on = charger_is_on

        service_data = {"entity_id": charger_ent}
        device_name = device.name

        if battery <= device.min_level and not charger_is_on:
            self._log_action(
                device_name,
                logging.WARNING,
                "[EmergencyCharge] %s below minimum level %.1f%% -> starting charging immediately (%s)",
                device_name,
                device.min_level,
                charger_ent,
            )
            await self.hass.services.async_call("switch", "turn_on", service_data, blocking=False)
            return True

        if precharge_required and not charger_is_on:
            self._log_action(
                device_name,
                logging.INFO,
                "[Precharge] %s below precharge level %.1f%% (current %.1f%%) -> activating charger (%s)",
                device_name,
                device.precharge_level,
                battery,
                charger_ent,
            )
            await self.hass.services.async_call("switch", "turn_on", service_data, blocking=False)
            return True

        if not is_home and charger_is_on:
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStop] %s not at home -> deactivating charger (%s)",
                device_name,
                charger_ent,
            )
            await self.hass.services.async_call("switch", "turn_off", service_data, blocking=False)
            return False

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
            await self.hass.services.async_call("switch", "turn_on", service_data, blocking=False)
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
            await self.hass.services.async_call("switch", "turn_off", service_data, blocking=False)
            return False

        if (
            charger_is_on
            and smart_start_active
            and start_time
            and now_local < start_time
            and not precharge_required
        ):
            self._log_action(
                device_name,
                logging.INFO,
                "[Precharge] %s reached the safety level and waits for the start at %s -> pausing charger (%s)",
                device_name,
                start_time.isoformat(),
                charger_ent,
            )
            await self.hass.services.async_call("switch", "turn_off", service_data, blocking=False)
            return False

        return expected_on

    def _log_action(self, device_name: str, level: int, message: str, *args: Any) -> None:
        rendered = message % args if args else message
        if self._last_action_log.get(device_name) == rendered:
            _LOGGER.debug("Skipping duplicate action log for %s: %s", device_name, rendered)
            return
        self._last_action_log[device_name] = rendered
        _LOGGER.log(level, rendered)
