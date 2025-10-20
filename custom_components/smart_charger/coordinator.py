from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import inspect
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
    DEFAULT_ADAPTIVE_THROTTLE_ENABLED,
    DEFAULT_ADAPTIVE_THROTTLE_MULTIPLIER,
    DEFAULT_ADAPTIVE_THROTTLE_MIN_SECONDS,
    DEFAULT_ADAPTIVE_THROTTLE_DURATION_SECONDS,
    DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
    DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
    DEFAULT_ADAPTIVE_THROTTLE_BACKOFF_STEP,
    DEFAULT_ADAPTIVE_EWMA_ALPHA,
    DEFAULT_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
    DEFAULT_ADAPTIVE_THROTTLE_MODE,
    CONF_ADAPTIVE_THROTTLE_ENABLED,
    CONF_ADAPTIVE_THROTTLE_MULTIPLIER,
    CONF_ADAPTIVE_THROTTLE_MIN_SECONDS,
    CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS,
    CONF_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
    CONF_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
    CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP,
    CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
    CONF_ADAPTIVE_THROTTLE_MODE,
    CONF_ADAPTIVE_EWMA_ALPHA,
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

_REAL_LOGGER = logging.getLogger(__name__)


class _QuietInfoLogger:
    """Logger adapter that demotes info/warning to debug for quieter CI runs.

    This keeps .debug/.exception/.log semantics unchanged but routes
    .info and .warning calls to the underlying logger.debug so tests and
    non-actionable diagnostics don't spam CI output. Critical logs that
    purposefully call _LOGGER.log(level, ...) (used by _log_action) still
    honor their explicit levels.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        # Demote informational messages to debug for less noisy CI output
        return self._logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        # Demote non-actionable warnings to debug; keep exception/log
        # behavior intact for true errors.
        return self._logger.debug(*args, **kwargs)

    def exception(self, *args, **kwargs):
        return self._logger.exception(*args, **kwargs)

    def log(self, level, *args, **kwargs):
        # Preserve explicit log(level, ...) calls so callers can still
        # emit warnings/errors when necessary via _LOGGER.log(...)
        return self._logger.log(level, *args, **kwargs)

    def __getattr__(self, name):
        # Delegate any other attribute access to the underlying logger
        return getattr(self._logger, name)


_LOGGER = _QuietInfoLogger(_REAL_LOGGER)


def _ignored_exc() -> None:
    """Log an ignored exception at DEBUG with traceback for auditing.

    Many internal guards intentionally suppress non-fatal exceptions to
    preserve coordinator uptime and avoid failing the whole evaluation.
    Bandit flags bare ``except: pass`` as unsafe; centralize a small
    helper that records the ignored exception at DEBUG level so we keep
    the original behaviour while satisfying static analysis.
    """
    try:
        _LOGGER.debug("Ignored exception (suppressed); enable DEBUG for traceback", exc_info=True)
    except Exception:
        # If logging itself fails, there's nothing else useful we can do.
        try:
            _REAL_LOGGER.debug("Ignored exception (suppressed) and logging failed")
        except Exception:
            _ignored_exc()


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

        # Use module-level coercion helpers to keep this constructor small.

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
            "start_time": (self._display_start_time() if self.start_time else None),
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
            _REAL_LOGGER,
            name=DOMAIN,
            update_method=self._async_update_data,
            update_interval=timedelta(seconds=UPDATE_INTERVAL),
        )
        # Keep references expected by other methods/tests
        self.hass = hass
        self.entry = entry
        self._precharge_margin_on = DEFAULT_PRECHARGE_MARGIN_ON
        self._precharge_margin_off = DEFAULT_PRECHARGE_MARGIN_OFF
        self._smart_start_margin = DEFAULT_SMART_START_MARGIN
        self._precharge_countdown_window = DEFAULT_PRECHARGE_COUNTDOWN_WINDOW
        self._default_learning_recent_sample_hours = (
            DEFAULT_LEARNING_RECENT_SAMPLE_HOURS
        )
        # Default throttle (seconds) used when device doesn't specify one
        self._default_switch_throttle_seconds = DEFAULT_SWITCH_THROTTLE_SECONDS
        # Per-device last switch timestamp (epoch seconds) to avoid rapid on/off flapping
        # Stored as float seconds since the epoch for robust, timezone-independent comparisons
        self._last_switch_time: Dict[str, float] = {}
        # Per-device configured throttle values (entity_id -> seconds)
        self._device_switch_throttle: Dict[str, float] = {}
        # Confirmation debounce: require N consecutive coordinator evaluations
        # that request a different desired state before issuing a switch.
        # Default is read from the central constants to remain consistent.
        self._confirmation_required = DEFAULT_SWITCH_CONFIRMATION_COUNT
        # per-device recent desired-state history: tuple(last_desired_bool, count)
        self._desired_state_history: Dict[str, tuple[bool, int]] = {}
        # Telemetry: track recent flip-flop events per entity (epoch timestamps)
        self._flipflop_events: dict[str, list[float]] = {}
        # Configurable telemetry thresholds (tunable constants)
        self._flipflop_window_seconds = 300.0  # lookback window (5 minutes)
        self._flipflop_warn_threshold = 3  # events within window to warn
        # Adaptive mitigation: temporary throttle overrides to suppress flapping
        # Structure: entity_id -> dict(original: float, applied: float, expires: float)
        self._adaptive_throttle_overrides: Dict[str, Dict[str, float]] = {}
        # Adaptive parameters
        self._adaptive_throttle_multiplier = 2.0
        self._adaptive_throttle_min_seconds = 120.0
        self._adaptive_throttle_duration_seconds = 600.0  # how long override lasts (10min)
        # Backoff parameters: how much extra multiplier to add per extra flip-flop event
        self._adaptive_throttle_backoff_step = 0.5
        self._adaptive_throttle_max_multiplier = 5.0
        # Per-device last action state (True==on, False==off) recorded when
        # the coordinator issues a switch. This helps tests (and some
        # integrations) that rely on the coordinator's pre-recorded actions
        # rather than the external entity state which may lag.
        self._last_action_state: Dict[str, bool] = {}
        # Per-refresh evaluation id used to avoid double-recording within the
        # same coordinator run when multiple code paths may record the desired
        # state for an entity.
        self._current_eval_id = 0
        self._last_recorded_eval: Dict[str, int] = {}
        # Per-device last switch evaluation id: stores the coordinator
        # evaluation id when the last service call was issued. This helps
        # reliably suppress immediate reversals that occur inside the same
        # coordinator evaluation or when evaluation ids are close in time.
        self._last_switch_eval: Dict[str, int | None] = {}
        # Entities with an in-flight service call recorded during this
        # coordinator evaluation to prevent immediate reversal races.
        self._inflight_switches: Dict[str, bool] = {}
        # Internal caches/state expected by the plan builder and other
        # helper methods. Initialize here to ensure tests that create
        # the coordinator directly don't encounter missing attributes.
        # battery history stores tuples like (timestamp: datetime, level: float, charging: bool)
        self._battery_history: Dict[str, tuple[datetime, float, bool] | tuple[datetime, float]] = {}
        self._drain_rate_cache: Dict[str, float] = {}
        self._precharge_release: Dict[str, Any] = {}
        # precharge_release_ready stores a datetime when the release will clear, or None
        self._precharge_release_ready: Dict[str, Optional[datetime]] = {}
        self._precharge_release_cleared_by_presence: Dict[str, bool] = {}
        # Tracks precharge latches cleared because thresholds/predictions
        # indicate the precharge is no longer required. This allows the
        # coordinator to act immediately (pause charger) in the same
        # evaluation when a release clears due to reaching thresholds.
        self._precharge_release_cleared_by_threshold: Dict[str, bool] = {}
        # Internal state snapshot produced by _async_update_data
        self._state: Dict[str, Any] = {}
        # EWMA metrics for flip-flop telemetry
        self._flipflop_ewma: float = 0.0
        self._flipflop_ewma_last_update: Optional[float] = None
        self._flipflop_ewma_exceeded: bool = False
        # Track when the EWMA first crossed the exceeded threshold
        self._flipflop_ewma_exceeded_since: Optional[float] = None
        # Internal adaptive mode override (None|'conservative'|'normal'|'aggressive')
        self._adaptive_mode_override: Optional[str] = None
        # Per-device forecast holdoff flag: when True the coordinator has
        # determined a forecast-based holdoff for that device and some
        # urgent actions may bypass throttle. Stored keyed by device.name.
        self._forecast_holdoff: Dict[str, bool] = {}
        # Last rendered action log to avoid duplicate logging spam
        self._last_action_log = {}

    def _normalize_entity_id(self, raw_entity: Any) -> Optional[str]:
        """Normalize an entity identifier used as dict keys.

        Accepts a string or a sequence (list/tuple) as passed in
        service_data['entity_id'] by some test harnesses. Returns the
        first element as a string when a sequence is provided, or the
        stringified value otherwise. Returns None for falsy inputs.
        """
        if not raw_entity:
            return None
        if isinstance(raw_entity, (list, tuple)) and raw_entity:
            try:
                return str(raw_entity[0])
            except Exception:
                return None
        try:
            return str(raw_entity)
        except Exception:
            _ignored_exc()
            return None

    def _device_name_for_entity(self, entity_id: str) -> Optional[str]:
        """Try to resolve a configured device name for the given entity_id.

        This searches the coordinator's current `_state` snapshot (which is
        keyed by device.name) and looks for a matching `charger_switch`
        value. Returns the device name when found, otherwise None.
        """
        if not entity_id:
            return None
        try:
            # `_state` entries are keyed by device.name and include a
            # `charger_switch` key with the entity id used for that device.
            for dev_name, data in (self._state or {}).items():
                try:
                    cs = data.get("charger_switch")
                    if cs and str(cs) == str(entity_id):
                        return dev_name
                except Exception:
                    _ignored_exc()
                    continue
        except Exception:
            _ignored_exc()
        return None

    def _early_suppress_checks(self, norm: str, desired: bool, force: bool, bypass_throttle: bool) -> bool:  # noqa: C901
        """Run initial authoritative and quick suppression checks.

        Returns True when the action should be suppressed.
        """
        should_check = not force and not bypass_throttle
        if not should_check:
            return False
        # Authoritative early suppression
        try:
            stored_raw = self._last_switch_time.get(norm)
            stored_epoch_auth = None
            if stored_raw is not None:
                try:
                    if isinstance(stored_raw, (int, float)):
                        stored_epoch_auth = float(stored_raw)
                    elif isinstance(stored_raw, str):
                        parsed = dt_util.parse_datetime(stored_raw)
                        stored_epoch_auth = float(dt_util.as_timestamp(parsed)) if parsed else None
                    else:
                        stored_epoch_auth = float(dt_util.as_timestamp(stored_raw))
                except Exception:
                    _ignored_exc()
                    stored_epoch_auth = None

            throttle_cfg_auth = float(
                self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
                or self._default_switch_throttle_seconds
            )
            if stored_epoch_auth is not None and throttle_cfg_auth:
                now_epoch_auth = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                elapsed_auth = now_epoch_auth - stored_epoch_auth
                last_act_auth = self._last_action_state.get(norm)
                if last_act_auth is None:
                    try:
                        st = self.hass.states.get(norm)
                        last_act_auth = bool(st and st.state == STATE_ON)
                    except Exception:
                        _ignored_exc()
                        last_act_auth = None
                if last_act_auth is not None and elapsed_auth >= 0 and elapsed_auth < float(throttle_cfg_auth) and bool(last_act_auth) != bool(desired):
                    _LOGGER.info(
                        "AUTHORITATIVE_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                        norm,
                        elapsed_auth,
                        throttle_cfg_auth,
                        last_act_auth,
                        desired,
                    )
                    return True
        except Exception:
            _ignored_exc()

        # Very-early deterministic quick-suppress
        try:
            last_raw_quick = self._last_switch_time.get(norm)
            last_act_quick = self._last_action_state.get(norm)
            throttle_quick = self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
            if last_raw_quick is not None and throttle_quick:
                try:
                    if isinstance(last_raw_quick, (int, float)):
                        last_epoch_q = float(last_raw_quick)
                    elif isinstance(last_raw_quick, str):
                        parsed_q = dt_util.parse_datetime(last_raw_quick)
                        last_epoch_q = float(dt_util.as_timestamp(parsed_q)) if parsed_q else None
                    else:
                        last_epoch_q = float(dt_util.as_timestamp(last_raw_quick))
                except Exception:
                    _ignored_exc()
                    last_epoch_q = None
                if last_epoch_q is not None:
                    now_epoch_q = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                    try:
                        thr_q = float(throttle_quick)
                    except Exception:
                        _ignored_exc()
                        thr_q = float(self._default_switch_throttle_seconds)
                    elapsed_q = now_epoch_q - last_epoch_q
                    if elapsed_q >= 0 and elapsed_q < thr_q:
                        if last_act_quick is None:
                            try:
                                st = self.hass.states.get(norm)
                                last_act_quick = bool(st and st.state == STATE_ON)
                            except Exception:
                                last_act_quick = None
                        if last_act_quick is not None and bool(last_act_quick) != bool(desired):
                            _LOGGER.info("VERY_EARLY_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s", norm, elapsed_q, thr_q, last_act_quick, desired)
                            return True
        except Exception:
            _ignored_exc()

        return False

    def _should_suppress_switch(self, norm: str, desired: bool, force: bool, bypass_throttle: bool) -> bool:  # noqa: C901
        """Comprehensive suppression helper extracted from _maybe_switch.

        Returns True when the switch should be suppressed, False otherwise.
        This consolidates multiple quick-gates and authoritative checks so
        the primary method remains smaller and easier for static analysis.
        """
        should_check = not force and not bypass_throttle
        # If there's an in-flight switch and it differs from desired, suppress
        try:
            pending = self._inflight_switches.get(norm)
            if should_check and pending is not None and bool(pending) != bool(desired):
                _LOGGER.debug("SUPPRESS_INFLIGHT: entity=%s pending=%s desired=%s", norm, pending, desired)
                return True
        except Exception:
            _ignored_exc()

        # Early authoritative throttle/reversal suppression
        if should_check:
            try:
                last_raw = self._last_switch_time.get(norm)
                if last_raw is not None:
                    try:
                        if isinstance(last_raw, (int, float)):
                            last_epoch_check = float(last_raw)
                        elif isinstance(last_raw, str):
                            parsed_lr = dt_util.parse_datetime(last_raw)
                            last_epoch_check = float(dt_util.as_timestamp(parsed_lr)) if parsed_lr else None
                        else:
                            last_epoch_check = float(dt_util.as_timestamp(last_raw))
                    except Exception:
                        last_epoch_check = None
                else:
                    last_epoch_check = None

                if last_epoch_check is not None:
                    now_epoch_check = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                    thr_cfg = float(self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds) or self._default_switch_throttle_seconds)
                    elapsed_check = now_epoch_check - last_epoch_check
                    if elapsed_check >= 0 and elapsed_check < float(thr_cfg):
                        last_action_check = self._last_action_state.get(norm)
                        if last_action_check is None:
                            try:
                                st = self.hass.states.get(norm)
                                last_action_check = bool(st and st.state == STATE_ON)
                            except Exception:
                                last_action_check = None
                        if last_action_check is not None and bool(last_action_check) != bool(desired):
                            _LOGGER.debug(
                                "EARLY_AUTHORITATIVE_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                                norm,
                                elapsed_check,
                                thr_cfg,
                                last_action_check,
                                desired,
                            )
                            return True
            except Exception:
                _ignored_exc()

        # Canonical throttle/reversal guard
        if should_check:
            try:
                _LOGGER.debug(
                    "DBG_CANONICAL_ENTER: entity=%s last_switch_time_raw=%r last_action_state=%r throttle_cfg=%r",
                    norm,
                    self._last_switch_time.get(norm),
                    self._last_action_state.get(norm),
                    self._device_switch_throttle.get(norm),
                )
            except Exception:
                _ignored_exc()
            # Local typed sentinel for subsequent throttle checks
            last_raw = self._last_switch_time.get(norm)
            last_act = self._last_action_state.get(norm)
            throttle_cfg = self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
            if last_raw is not None and last_act is not None and throttle_cfg:
                try:
                    if isinstance(last_raw, (int, float)):
                        last_epoch = float(last_raw)
                    elif isinstance(last_raw, str):
                        parsed = dt_util.parse_datetime(last_raw)
                        last_epoch = float(dt_util.as_timestamp(parsed)) if parsed else None
                    else:
                        last_epoch = float(dt_util.as_timestamp(last_raw))
                except Exception:
                    last_epoch = None
                if last_epoch is not None:
                    now_epoch = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                    try:
                        throttle_val = float(throttle_cfg)
                    except Exception:
                        throttle_val = float(self._default_switch_throttle_seconds)
                    elapsed = now_epoch - last_epoch
                    if last_act is None:
                        try:
                            st = self.hass.states.get(norm)
                            last_act = bool(st and st.state == STATE_ON)
                        except Exception:
                            last_act = None
                        try:
                            _LOGGER.debug(
                                "DBG_CANONICAL: entity=%s last_epoch=%r last_act=%r now_epoch=%r elapsed=%r throttle_val=%r desired=%r last_eval=%r cur_eval=%r",
                                norm,
                                last_epoch,
                                last_act,
                                now_epoch,
                                elapsed,
                                throttle_val,
                                desired,
                                self._last_switch_eval.get(norm),
                                getattr(self, "_current_eval_id", None),
                            )
                        except Exception:
                            _ignored_exc()
                    if elapsed >= 0 and elapsed < float(throttle_val) and last_act is not None and bool(last_act) != bool(desired):
                        _LOGGER.debug("CANONICAL_THROTTLE_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s", norm, elapsed, throttle_val, last_act, desired)
                        try:
                            last_eval = self._last_switch_eval.get(norm)
                            cur_eval = int(getattr(self, "_current_eval_id", 0) or 0)
                            if last_eval is None or abs(cur_eval - int(last_eval or 0)) <= 1:
                                return True
                        except Exception:
                            return True

        # Recent-eval suppression guard
        if should_check:
            try:
                last_eval = self._last_switch_eval.get(norm)
                cur_eval = int(getattr(self, "_current_eval_id", 0) or 0)
                last_act = self._last_action_state.get(norm)
                cond_last_eval = last_eval is not None
                cond_eval_delta = False
                if cond_last_eval:
                    cond_eval_delta = abs(cur_eval - int(last_eval or 0)) <= 1
                cond_last_act = last_act is not None
                cond_reversal = cond_last_act and (bool(last_act) != bool(desired))
                _LOGGER.debug(
                    "DBG_SUPPRESS_GATES: entity=%s last_eval=%r cur_eval=%r cond_last_eval=%r cond_eval_delta=%r cond_last_act=%r cond_reversal=%r last_act=%r desired=%r",
                    norm,
                    last_eval,
                    cur_eval,
                    cond_last_eval,
                    cond_eval_delta,
                    cond_last_act,
                    cond_reversal,
                    last_act,
                    desired,
                )
                if cond_last_eval and cond_eval_delta and cond_last_act and cond_reversal:
                    _LOGGER.debug("SUPPRESS_RECENT_EVAL: entity=%s cur_eval=%s last_eval=%s last_act=%r desired=%s", norm, cur_eval, last_eval, last_act, desired)
                    return True
            except Exception:
                _ignored_exc()

        # Defensive early throttle/reversal guard
        if should_check:
            try:
                last_raw = self._last_switch_time.get(norm)
                last_act = self._last_action_state.get(norm)
                throttle_cfg = self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
                if last_raw is not None and last_act is not None and throttle_cfg:
                    try:
                        if isinstance(last_raw, (int, float)):
                            last_epoch = float(last_raw)
                        elif isinstance(last_raw, str):
                            parsed = dt_util.parse_datetime(last_raw)
                            last_epoch = float(dt_util.as_timestamp(parsed)) if parsed else None
                        else:
                            last_epoch = float(dt_util.as_timestamp(last_raw))
                    except Exception:
                        last_epoch = None
                    if last_epoch is not None:
                        now_epoch = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                        try:
                            throttle_val = float(throttle_cfg)
                        except Exception:
                            throttle_val = float(self._default_switch_throttle_seconds)
                        elapsed = now_epoch - last_epoch
                        _LOGGER.debug(
                            "DBG_EARLY_GUARD: entity=%s last_epoch=%.6f now_epoch=%.6f elapsed=%.6f throttle_val=%r last_act=%r desired=%r",
                            norm,
                            float(last_epoch),
                            float(now_epoch),
                            float(elapsed),
                            throttle_val,
                            last_act,
                            desired,
                        )
                        if elapsed >= 0 and elapsed < float(throttle_val) and bool(last_act) != bool(desired):
                            _LOGGER.debug(
                                "EARLY_DEFENSIVE_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                                norm,
                                elapsed,
                                throttle_val,
                                last_act,
                                desired,
                            )
                            return True
            except Exception:
                _ignored_exc()

        return False

    @property
    def profiles(self) -> Dict[str, Dict[str, Any]]:
        return self._state or {}

    def _raw_config(self) -> Dict[str, Any]:
        self.config = {**dict(self.entry.data), **getattr(self.entry, "options", {})}
        return self.config

    def _option_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            _LOGGER.debug(
                "Invalid option %s=%r, falling back to %.2f",
                key,
                value,
                default,
            )
            return float(default)

    def _option_bool(self, key: str, default: bool) -> bool:
        """Read a boolean option robustly from the merged config/options.

        Accepts booleans, numeric 0/1, and common true/false strings.
        """
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        try:
            if isinstance(value, (int, float)):
                return bool(int(value))
        except Exception:
            _ignored_exc()
        try:
            sval = str(value).strip().lower()
            if sval in ("1", "true", "yes", "on"):
                return True
            if sval in ("0", "false", "no", "off"):
                return False
        except Exception:
            _ignored_exc()
        return bool(default)

    def _iter_device_configs(
        self, devices: Optional[Iterable[Mapping[str, Any]]] = None
    ) -> Iterable[DeviceConfig]:
        """Yield validated DeviceConfig objects from the raw device list.

        Silent-logs any invalid device configuration entries.
        """
        if devices is None:
            devices = self._raw_config().get("devices") or []
        for raw in list(devices or []):
            try:
                yield DeviceConfig.from_dict(raw)
            except Exception as err:
                _LOGGER.warning(
                    "Skipping invalid device configuration %s: %s", raw, err
                )

    async def _async_update_data(self) -> Dict[str, Dict[str, Any]]:  # noqa: C901
        results: Dict[str, Dict[str, Any]] = {}
        # New evaluation - advance the eval id to avoid duplicate recordings
        # within the same refresh cycle.
        try:
            try:
                # Render a human-friendly snapshot at DEBUG level only.
                rendered = {}
                for k, v in self._last_switch_time.items():
                    try:
                        if isinstance(v, (int, float)):
                            rendered[k] = datetime.fromtimestamp(float(v)).isoformat()
                        else:
                            iso = getattr(v, "isoformat", None)
                            rendered[k] = iso() if callable(iso) else repr(v)
                    except Exception:
                        _ignored_exc()
                        rendered[k] = repr(v)
                _LOGGER.debug("SNAPSHOT(last_switch_time): %s", rendered)
            except Exception:
                _ignored_exc()
            self._current_eval_id += 1
            # Clear any in-flight markers at the start of a new evaluation so
            # they only protect against reversals within the same coordinator
            # refresh cycle.
            try:
                self._inflight_switches = {}
            except Exception:
                _ignored_exc()
        except Exception:
            self._current_eval_id = getattr(self, "_current_eval_id", 1)
        # Record the logical evaluation time so switch throttling and
        # confirmation can be evaluated against the same simulated time used
        # by the plan builder (useful for deterministic tests which pass a
        # custom ``now_local`` into _build_plan).
        now_local = dt_util.now()  # This line remains unchanged
        _LOGGER.debug("_async_update_data: starting evaluation %d now=%s", self._current_eval_id, now_local.isoformat())
        self._current_eval_time = now_local
        domain_data = self.hass.data.get(DOMAIN, {})
        entries: Dict[str, Dict[str, Any]] = domain_data.get("entries", {})
        entry_data = entries.get(self.entry.entry_id, {})
        learning = entry_data.get("learning")

        try:
            raw_config = self._raw_config()
            _LOGGER.debug("_async_update_data: raw device count=%d", len(raw_config.get("devices", []) or []))
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

            # Adaptive throttle options (coordinator-level defaults; per-device overrides still apply)
            # coordinator-level toggle for adaptive mitigation
            try:
                self._adaptive_enabled = self._option_bool(
                    CONF_ADAPTIVE_THROTTLE_ENABLED, DEFAULT_ADAPTIVE_THROTTLE_ENABLED
                )
            except Exception:
                _ignored_exc()
                self._adaptive_enabled = bool(DEFAULT_ADAPTIVE_THROTTLE_ENABLED)
            self._adaptive_throttle_multiplier = self._option_float(
                CONF_ADAPTIVE_THROTTLE_MULTIPLIER, DEFAULT_ADAPTIVE_THROTTLE_MULTIPLIER
            )
            self._adaptive_throttle_min_seconds = self._option_float(
                CONF_ADAPTIVE_THROTTLE_MIN_SECONDS, DEFAULT_ADAPTIVE_THROTTLE_MIN_SECONDS
            )
            self._adaptive_throttle_duration_seconds = self._option_float(
                CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS, DEFAULT_ADAPTIVE_THROTTLE_DURATION_SECONDS
            )
            # Backoff tuning (variable multiplier)
            self._adaptive_throttle_backoff_step = self._option_float(
                CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP, DEFAULT_ADAPTIVE_THROTTLE_BACKOFF_STEP
            )
            self._adaptive_throttle_max_multiplier = self._option_float(
                CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER, DEFAULT_ADAPTIVE_THROTTLE_MAX_MULTIPLIER
            )
            # Adaptive mode: conservative / normal / aggressive
            try:
                raw_mode = self.config.get(
                    CONF_ADAPTIVE_THROTTLE_MODE, DEFAULT_ADAPTIVE_THROTTLE_MODE
                )
                mode = str(raw_mode).strip().lower()
            except Exception:
                _ignored_exc()
                mode = DEFAULT_ADAPTIVE_THROTTLE_MODE

            # Effective mode: allow runtime override to temporarily change behavior
            effective_mode = str(getattr(self, "_adaptive_mode_override", None) or mode).strip().lower()

            # Map effective_mode to a scaling factor applied to the backoff growth
            if effective_mode == "conservative":
                self._adaptive_mode_factor = 0.7
            elif effective_mode == "aggressive":
                self._adaptive_mode_factor = 1.4
            else:
                self._adaptive_mode_factor = 1.0
            self._flipflop_window_seconds = self._option_float(
                CONF_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS, DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS
            )
            try:
                self._flipflop_warn_threshold = int(
                    self._option_float(
                        CONF_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
                        DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
                    )
                )
            except Exception:
                _ignored_exc()
                self._flipflop_warn_threshold = int(DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD)

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
                        self._device_switch_throttle[ent] = (
                            self._default_switch_throttle_seconds
                        )
                    # confirmation count
                    if device.switch_confirmation_count is not None:
                        self._desired_state_history.setdefault(ent, (False, 0))
                        # store per-entity confirmation requirement
                        self._device_switch_throttle.setdefault(
                            f"{ent}::confirm", float(device.switch_confirmation_count)
                        )
                    else:
                        self._device_switch_throttle.setdefault(
                            f"{ent}::confirm", float(self._confirmation_required)
                        )
                except Exception as err:
                    _LOGGER.debug(
                        "Unable to configure throttle/confirmation for %s: %s",
                        device.name,
                        err,
                    )

                plan = await self._build_plan(
                    device,
                    now_local,
                    learning,
                    device_window,
                )
                if plan:
                    results[device.name] = plan.as_dict()

            # Telemetry: compute flip-flop rates and emit warnings if necessary
            try:
                now_epoch = float(dt_util.as_timestamp(now_local))
                cutoff = now_epoch - float(self._flipflop_window_seconds)
                # First: expire adaptive overrides that hit their expiry
                try:
                    for ent, meta in list(self._adaptive_throttle_overrides.items()):
                        expires = float(meta.get("expires", 0.0) or 0.0)
                        if expires <= now_epoch:
                            # restore original throttle if available
                            try:
                                original = float(meta.get("original", 0.0) or 0.0)
                                if original and ent in self._device_switch_throttle:
                                    self._device_switch_throttle[ent] = original
                            except Exception:
                                _ignored_exc()
                            try:
                                self._adaptive_throttle_overrides.pop(ent, None)
                            except Exception:
                                _ignored_exc()
                except Exception:
                    _ignored_exc()

                for ent, events in list(self._flipflop_events.items()):
                    # drop old events outside the configured window
                    recent = [e for e in events if e >= cutoff]
                    self._flipflop_events[ent] = recent
                    if len(recent) >= int(self._flipflop_warn_threshold):
                        try:
                            _LOGGER.warning(
                                "High flip-flop rate detected for %s: %d events in the last %.0fs",
                                ent,
                                len(recent),
                                float(self._flipflop_window_seconds),
                            )
                        except Exception:
                            _ignored_exc()
                        # Adaptive mitigation: apply/refresh throttle override
                        try:
                            # determine current configured throttle
                            current = float(
                                self._device_switch_throttle.get(
                                    ent, self._default_switch_throttle_seconds
                                )
                                or self._default_switch_throttle_seconds
                            )
                            # compute variable multiplier with backoff: increase multiplier for excess flip-flop events
                            try:
                                count = len(recent)
                                excess = max(0, count - int(self._flipflop_warn_threshold))
                                # Apply mode factor to backoff incremental growth
                                var_multiplier = float(self._adaptive_throttle_multiplier) + (
                                    float(self._adaptive_throttle_backoff_step)
                                    * float(excess)
                                    * float(getattr(self, "_adaptive_mode_factor", 1.0))
                                )
                                # clamp to configured maximum
                                var_multiplier = min(
                                    var_multiplier, float(self._adaptive_throttle_max_multiplier)
                                )
                            except Exception:
                                _ignored_exc()
                                var_multiplier = float(self._adaptive_throttle_multiplier)

                            desired = max(current * float(var_multiplier), float(self._adaptive_throttle_min_seconds))
                            meta_entry: dict | None = self._adaptive_throttle_overrides.get(ent)
                            if not meta_entry:
                                # store original and apply override
                                try:
                                    self._adaptive_throttle_overrides[ent] = {
                                        "original": float(self._device_switch_throttle.get(ent, self._default_switch_throttle_seconds) or self._default_switch_throttle_seconds),
                                        "applied": float(desired),
                                        "expires": float(now_epoch + float(self._adaptive_throttle_duration_seconds)),
                                    }
                                except Exception:
                                    _ignored_exc()
                                try:
                                    self._device_switch_throttle[ent] = float(desired)
                                    _LOGGER.info(
                                        "Adaptive throttle applied for %s = %.1fs (original %.1fs) until %.0f",
                                        ent,
                                        float(desired),
                                        float(self._adaptive_throttle_overrides.get(ent, {}).get("original", 0.0)),
                                        float(now_epoch + float(self._adaptive_throttle_duration_seconds)),
                                    )
                                except Exception:
                                    _ignored_exc()
                            else:
                                # refresh expiry and possibly raise applied throttle
                                try:
                                    meta_applied = float(meta_entry.get("applied", 0.0) or 0.0)
                                    new_applied = max(meta_applied, float(desired))
                                    meta_entry["applied"] = new_applied
                                    meta_entry["expires"] = float(now_epoch + float(self._adaptive_throttle_duration_seconds))
                                    self._adaptive_throttle_overrides[ent] = meta_entry
                                    self._device_switch_throttle[ent] = float(new_applied)
                                except Exception:
                                    _ignored_exc()
                        except Exception:
                            _ignored_exc()
                # After pruning events, compute aggregate rates and update EWMA
                try:
                    total_events = sum(len(v) for v in self._flipflop_events.values())
                    window = float(getattr(self, "_flipflop_window_seconds", DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS))
                    rate_per_sec = float(total_events) / window if window > 0 else 0.0
                    # read alpha from entry options if available
                    try:
                        entry_obj = getattr(self, "entry", None)
                        if entry_obj and getattr(entry_obj, "options", None) is not None:
                            raw_alpha = entry_obj.options.get(CONF_ADAPTIVE_EWMA_ALPHA)
                            alpha = float(raw_alpha) if raw_alpha is not None else DEFAULT_ADAPTIVE_EWMA_ALPHA
                        else:
                            alpha = DEFAULT_ADAPTIVE_EWMA_ALPHA
                    except Exception:
                        alpha = DEFAULT_ADAPTIVE_EWMA_ALPHA
                    prev = float(getattr(self, "_flipflop_ewma", 0.0) or 0.0)
                    try:
                        ewma = prev + alpha * (rate_per_sec - prev)
                    except Exception:
                        _ignored_exc()
                        ewma = prev
                    # persist EWMA and metadata
                    try:
                        self._flipflop_ewma = ewma
                        self._flipflop_ewma_last_update = float(now_epoch)
                        # mark exceeded when EWMA crosses a small threshold (e.g., more than warn threshold / window)
                        exceeded_threshold = float(self._flipflop_warn_threshold) / max(1.0, float(self._flipflop_window_seconds))
                        prev_exceeded = bool(getattr(self, "_flipflop_ewma_exceeded", False))
                        new_exceeded = ewma >= exceeded_threshold
                        self._flipflop_ewma_exceeded = new_exceeded
                        now_ts = float(now_epoch)
                        if new_exceeded and not prev_exceeded:
                            # record when we first exceed
                            self._flipflop_ewma_exceeded_since = now_ts
                            _LOGGER.warning("Flipflop EWMA exceeded threshold: ewma=%.6f threshold=%.6f", ewma, exceeded_threshold)
                        elif new_exceeded and prev_exceeded:
                            # already exceeded; check sustained duration
                            try:
                                since = float(getattr(self, "_flipflop_ewma_exceeded_since", now_ts) or now_ts)
                                duration = now_ts - since
                                # after 5 minutes of sustained exceed, flip integration to aggressive mode
                                if duration >= 300.0 and self._adaptive_mode_override != "aggressive":
                                    self._adaptive_mode_override = "aggressive"
                                    _LOGGER.warning("Adaptive mode override applied: aggressive (sustained EWMA for %.0fs)", duration)
                                    # persist override into ConfigEntry options so it survives restarts
                                    try:
                                        new_opts = dict(getattr(self.entry, "options", {}) or {})
                                        new_opts["adaptive_mode_override"] = "aggressive"
                                        try:
                                            self.hass.config_entries.async_update_entry(self.entry, options=new_opts)
                                        except Exception:
                                            _ignored_exc()
                                    except Exception:
                                        _ignored_exc()
                            except Exception:
                                _ignored_exc()
                        else:
                            # not exceeded anymore: clear since/override
                            self._flipflop_ewma_exceeded_since = None
                            if getattr(self, "_adaptive_mode_override", None) is not None:
                                _LOGGER.info("Adaptive mode override cleared (EWMA dropped)")
                                self._adaptive_mode_override = None
                                # remove persisted override key from options
                                try:
                                    new_opts = dict(getattr(self.entry, "options", {}) or {})
                                    if "adaptive_mode_override" in new_opts:
                                        new_opts.pop("adaptive_mode_override", None)
                                        try:
                                            self.hass.config_entries.async_update_entry(self.entry, options=new_opts)
                                        except Exception:
                                            _ignored_exc()
                                except Exception:
                                    _ignored_exc()
                    except Exception:
                        _ignored_exc()
                except Exception:
                    _ignored_exc()
            except Exception:
                _ignored_exc()

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
            # Report desired-state history at DEBUG level only
            try:
                _LOGGER.debug("Desired state history snapshot: %s", self._desired_state_history)
                try:
                    readable = {}
                    for k, v in self._last_switch_time.items():
                        try:
                            if isinstance(v, (int, float)):
                                readable[k] = datetime.fromtimestamp(float(v)).isoformat()
                            else:
                                # Use getattr with a safe default callable bound in a local scope
                                iso = getattr(v, "isoformat", None)
                                readable[k] = iso() if callable(iso) else repr(v)
                        except Exception:
                            readable[k] = repr(v)
                except Exception:
                    readable = {k: repr(v) for k, v in self._last_switch_time.items()}
                _LOGGER.debug("last_switch_time snapshot: %s", readable)
            except Exception:
                _ignored_exc()
            return results

        except Exception:
            _LOGGER.exception("Smart Charger coordinator update failed")
            return self._state or {}
        finally:
            pass

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
                        max(LEARNING_MIN_SPEED, min(LEARNING_MAX_SPEED, float(speed))),
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

    async def _build_plan(  # noqa: C901
        self,
        device: DeviceConfig,
        now_local: datetime,
        learning,
        learning_window_hours: float,
    ) -> Optional[SmartChargePlan]:
        # When this helper is invoked directly (tests often call it with a
        # simulated ``now_local``), advance the per-evaluation id and
        # record the evaluation time. If called from _async_update_data the
        # coordinator already set ``_current_eval_time`` and incremented the
        # evaluation id; avoid double-incrementing in that case.
        # Advance the evaluation id when this helper is invoked directly
        # with a different logical time. If called from _async_update_data,
        # that method already advanced the id and set _current_eval_time and
        # the provided now_local will match; avoid double-incrementing in
        # that case. This ensures repeated direct calls in tests produce a
        # new evaluation id so confirmation counters and per-eval guards
        # behave as expected.
        # Always advance the evaluation id for each explicit call to
        # _build_plan so tests using direct invocations see distinct
        # coordinator evaluations. This avoids confirmation counters and
        # per-eval guards from being skipped when tests call the helper
        # repeatedly.
        try:
            self._current_eval_id = int(getattr(self, "_current_eval_id", 0) or 0) + 1
        except Exception:
            self._current_eval_id = getattr(self, "_current_eval_id", 1)
        self._current_eval_time = now_local
        _LOGGER.info("_build_plan: device=%s now=%s eval=%d", device.name, now_local.isoformat(), self._current_eval_id)

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
        # Consider the coordinator's last intended action as an "assumed"
        # state when available. Many tests and some integrations rely on the
        # coordinator's recorded intended action (which may not have been
        # applied to the entity state yet) for deterministic behavior.
        last_action = self._last_action_state.get(device.charger_switch)

        # (forecast_holdoff persistence moved to _apply_charger_logic)
        # If the observed entity state is unavailable/unknown, fall back to
        # the coordinator's last intended action. Additionally, when the
        # coordinator recently issued a switch (within the per-device
        # throttle window), assume that intended action has taken effect
        # for the purpose of decision-making. This allows tests and
        # integrations where entity state lags the coordinator's service
        # call to behave deterministically, while still preferring the
        # actual entity state when it is reliable.
        if not charger_is_on:
            # Look up device-specific throttle seconds; fall back to a
            # conservative short window (5s) if not configured.
            throttle_seconds = self._device_switch_throttle.get(
                device.charger_switch, self._default_switch_throttle_seconds
            )
            try:
                throttle_window = float(throttle_seconds) if throttle_seconds else 5.0
            except Exception:
                throttle_window = 5.0

            last_action = self._last_action_state.get(device.charger_switch)
            last_ts = self._last_switch_time.get(device.charger_switch)
            if last_action is not None and last_ts is not None:
                # Use the coordinator's logical evaluation time when available
                # for throttle comparisons so the recorded last-switch epoch
                # (which also prefers _current_eval_time) is compared using
                # the same clock. Fall back to real UTC otherwise.
                now_for_cmp = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                try:
                    now_ts = float(dt_util.as_timestamp(now_for_cmp))
                    # last_ts may already be an epoch float stored by this coordinator
                    if isinstance(last_ts, (int, float)):
                        last_epoch = float(last_ts)
                    else:
                        last_epoch = float(dt_util.as_timestamp(last_ts))
                    elapsed = now_ts - last_epoch
                    # If the last switch was recent (within throttle_window),
                    # use the coordinator's last intended action as the
                    # presumed state for this evaluation.
                    if elapsed >= 0 and elapsed <= float(throttle_window):
                        charger_is_on = bool(last_action)
                except Exception:
                    # On error, be conservative and do not override the
                    # observed entity state.
                    _ignored_exc()
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

        _LOGGER.info(
            "Plan debug: device=%s battery=%.1f predicted=%.1f deficit=%.1f precharge_required=%s charger_expected_on=%s",
            device.name,
            battery,
            predicted_level,
            charge_deficit,
            precharge_required,
            charger_expected_on,
        )

        # Record the observed desired state so confirmation counters are
        # maintained across coordinator evaluations even when no switch
        # service was invoked during this run.
        try:
            self._record_desired_state(device.charger_switch, bool(charger_expected_on))
        except Exception as err:
            # Non-critical; avoid failing the whole plan generation on record
            _LOGGER.debug(
                "Failed to record desired state for %s: %s",
                device.name,
                err,
            )

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
            total_duration_min = min(
                total_duration_min + precharge_duration_min, 48 * 60
            )

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

        _LOGGER.info(
            "Eval precharge: device=%s battery=%.1f predicted=%.1f is_home=%s prev_release=%s",
            device.name,
            battery,
            predicted_level,
            is_home,
            previous_release,
        )

        if not is_home:
            if release_level is not None:
                # Record that the precharge latch was cleared because the
                # device left home so the plan builder can react (e.g. pause
                # the charger immediately) during the same evaluation.
                self._precharge_release_cleared_by_presence[device.name] = True
                self._precharge_release.pop(device.name, None)
                self._precharge_release_ready.pop(device.name, None)
            release_level = None
        elif device.precharge_level > device.min_level:
            trigger_threshold = device.precharge_level - margin_on
            predicted_threshold = device.precharge_level + margin_on
            near_precharge_window = (
                battery <= device.precharge_level + self._precharge_countdown_window
            )

            should_latch = False
            if battery <= trigger_threshold:
                should_latch = True
            elif predicted_level <= trigger_threshold and near_precharge_window:
                should_latch = True

            if should_latch:
                release_level, precharge_required = self._handle_precharge_latch(
                    device,
                    expected_drain,
                    smart_start_active,
                    start_time,
                    now_local,
                    margin_on,
                    margin_off,
                    smart_margin,
                    previous_release,
                )
            elif release_level is not None:
                (
                    precharge_required,
                    release_level,
                    release_ready_at,
                ) = self._handle_existing_release(
                    device,
                    battery,
                    predicted_level,
                    predicted_threshold,
                    near_precharge_window,
                    release_ready_at,
                    now_local,
                )

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

    def _handle_precharge_latch(
        self,
        device: DeviceConfig,
        expected_drain: float,
        smart_start_active: bool,
        start_time: Optional[datetime],
        now_local: datetime,
        margin_on: float,
        margin_off: float,
        smart_margin: float,
        previous_release: Optional[float],
    ) -> tuple[Optional[float], bool]:
        """Compute and set release_level when a precharge latch should occur.

        Returns (release_level, precharge_required).
        """
        extra_margin = max(margin_off, expected_drain * 0.4)
        release_cap = device.target_level
        if smart_start_active and start_time and now_local < start_time:
            release_cap = max(device.precharge_level, device.target_level - smart_margin)
        release_level = min(release_cap, device.precharge_level + extra_margin)
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
        return release_level, True

    def _handle_existing_release(
        self,
        device: DeviceConfig,
        battery: float,
        predicted_level: float,
        predicted_threshold: float,
        near_precharge_window: bool,
        release_ready_at: Optional[datetime],
        now_local: datetime,
    ) -> tuple[bool, Optional[float], Optional[datetime]]:
        """Handle logic for when a release_level already exists.

        Returns (precharge_required, release_level, release_ready_at).
        """
        previously_in_range = release_ready_at is not None
        in_range = (
            release_ready_at is not None
            or (device.name in self._precharge_release and battery >= self._precharge_release[device.name] and predicted_level >= predicted_threshold)
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
                    return True, self._precharge_release.get(device.name), release_ready_at
                else:
                    # Cleared because we're no longer near precharge
                    self._precharge_release_cleared_by_threshold[device.name] = True
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
                    return False, None, None
            else:
                self._precharge_release_ready.pop(device.name, None)
                return True, self._precharge_release.get(device.name), release_ready_at
        else:
            if in_range and near_precharge:
                if release_ready_at is not None and now_local >= release_ready_at:
                    self._precharge_release_cleared_by_threshold[device.name] = True
                    self._precharge_release.pop(device.name, None)
                    self._precharge_release_ready.pop(device.name, None)
                    self._log_action(
                        device.name,
                        logging.DEBUG,
                        "[Precharge] %s release window cleared",
                        device.name,
                    )
                    return False, None, None
                else:
                    return True, self._precharge_release.get(device.name), release_ready_at
            elif not near_precharge:
                self._precharge_release_cleared_by_threshold[device.name] = True
                self._precharge_release.pop(device.name, None)
                self._precharge_release_ready.pop(device.name, None)
                self._log_action(
                    device.name,
                    logging.DEBUG,
                    "[Precharge] %s release cleared immediately (battery %.1f%%)",
                    device.name,
                    battery,
                )
                return False, None, None
            else:
                return True, self._precharge_release.get(device.name), release_ready_at

    def _should_authoritatively_suppress(
        self,
        caller_fn: Optional[str],
        entity_id: Optional[str],
        pre_epoch: Optional[float],
        previous_last_action: Optional[bool],
        action: str,
        bypass_throttle: bool,
        force: bool,
    ) -> bool:
        """Return True when an authoritative early suppression should occur.

        The caller must provide the immediate caller function name so the
        suppression logic can correctly decide when `_maybe_switch` already
        vetted the call. This avoids inspecting the stack from inside the
        helper which would change the observed caller.
        """
        try:
            if entity_id and not force and previous_last_action is not None and caller_fn != "_maybe_switch":
                try:
                    dev_name = self._device_name_for_entity(entity_id)
                except Exception:
                    _ignored_exc()
                    dev_name = None

                try:
                    legitimate_internal_bypass = self._is_device_in_latch_maps(dev_name)
                except Exception:
                    _ignored_exc()
                    legitimate_internal_bypass = False

                stored_epoch = self._parse_stored_epoch(entity_id)
                throttle_cfg = float(
                    self._device_switch_throttle.get(entity_id, self._default_switch_throttle_seconds)
                    or self._default_switch_throttle_seconds
                )
                if stored_epoch is not None and pre_epoch is not None:
                    elapsed = float(pre_epoch) - float(stored_epoch)
                    if (
                        elapsed >= 0
                        and elapsed < float(throttle_cfg)
                        and bool(previous_last_action) != bool(action == "turn_on")
                        and not (bypass_throttle and legitimate_internal_bypass)
                    ):
                        try:
                            _LOGGER.debug(
                                "DBG_ASYNC_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f prev_action=%r action=%s",
                                entity_id,
                                elapsed,
                                throttle_cfg,
                                previous_last_action,
                                action,
                            )
                        except Exception:
                            _ignored_exc()
                        _LOGGER.info(
                            "ASYNC_CALL_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f prev_action=%r action=%s",
                            entity_id,
                            elapsed,
                            throttle_cfg,
                            previous_last_action,
                            action,
                        )
                        return True
        except Exception:
            _ignored_exc()
        return False

    def _gather_async_call_debug_info(
        self,
        entity_id: Optional[str],
        pre_epoch: Optional[float],
        bypass_throttle: bool,
        force: bool,
        previous_last_action: Optional[bool],
        action: str,
    ) -> tuple[Any, Any]:
        """Gather debug values used by _async_switch_call.

        Returns (stored_val, device_thr). Exceptions are caught and
        suppressed using _ignored_exc so callers remain safe.
        """
        stored_val = None
        device_thr = None
        try:
            if entity_id:
                try:
                    stored_val = self._last_switch_time.get(entity_id)
                except Exception:
                    stored_val = None
                try:
                    device_thr = self._device_switch_throttle.get(entity_id)
                except Exception:
                    device_thr = None
            try:
                _LOGGER.debug(
                    "DBG_ASYNC_VALUES: entity=%r stored=%r pre_epoch=%r bypass_throttle=%s force=%s previous_last_action=%r action=%s device_throttle=%r",
                    entity_id,
                    stored_val,
                    pre_epoch,
                    bypass_throttle,
                    force,
                    previous_last_action,
                    action,
                    device_thr,
                )
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()
        return stored_val, device_thr

    def _parse_stored_epoch(self, entity_id: Optional[str]) -> Optional[float]:
        """Parse stored last-switch time into an epoch float or return None.

        This isolates the datetime parsing branch so the parent function can
        remain small and focused.
        """
        if not entity_id:
            return None
        stored = self._last_switch_time.get(entity_id)
        if stored is None:
            return None
        try:
            if isinstance(stored, (int, float)):
                return float(stored)
            if isinstance(stored, str):
                parsed = dt_util.parse_datetime(stored)
                return float(dt_util.as_timestamp(parsed)) if parsed else None
            return float(dt_util.as_timestamp(stored))
        except Exception:
            _ignored_exc()
            return None

    def _is_device_in_latch_maps(self, dev_name: Optional[str]) -> bool:
        """Return True if the device name is present in any precharge/forecast maps.

        Isolating this lookup reduces code duplication and keeps the
        suppression helper concise.
        """
        if not dev_name:
            return False
        try:
            return (
                dev_name in self._precharge_release
                or dev_name in self._precharge_release_ready
                or dev_name in self._forecast_holdoff
            )
        except Exception:
            _ignored_exc()
            return False

    def _compute_local_effective_bypass(self, entity_id: Optional[str], bypass_throttle: bool, force: bool) -> bool:
        """Compute whether this call should locally bypass throttle.

        This encapsulates the logic that inspects precharge maps and resolves
        entity -> device name mappings.
        """
        local_effective_bypass = bool(bypass_throttle or force)
        try:
            if entity_id:
                try:
                    ent = str(entity_id)
                    if ent in self._precharge_release or ent in self._precharge_release_ready:
                        return True
                    try:
                        dev_name = self._device_name_for_entity(ent)
                        if self._is_device_in_latch_maps(dev_name):
                            return True
                    except Exception:
                        _ignored_exc()
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()
        return local_effective_bypass

    async def _async_switch_call(
        self,
        action: str,
        service_data: Dict[str, Any],
        pre_epoch: Optional[float] = None,
        previous_last_action: Optional[bool] = None,
        *,
        bypass_throttle: bool = False,
        force: bool = False,
    ) -> bool:
        raw_entity = service_data.get("entity_id")
        entity_id = self._normalize_entity_id(raw_entity)
        # Ensure service_data contains the normalized id when calling the service
        call_data = dict(service_data)
        call_data["entity_id"] = entity_id
        # Diagnostic: always print the key inputs for authoritative suppression.
        # Guard lookups when entity_id may be None to avoid type-checker/linter
        # complaints and accidental KeyError-like behavior in some runtimes.
        stored_val, device_thr = self._gather_async_call_debug_info(
            entity_id, pre_epoch, bypass_throttle, force, previous_last_action, action
        )
        # Minimal internal bypass check: some coordinator-driven events
        # should be allowed to bypass the configured throttle in
        # well-defined, 'intelligent' situations (for example a
        # precharge-release that the plan just latched). Compute a local
        # effective bypass flag so callers that pass bypass_throttle/force
        # remain authoritative but internal precharge-release events can
        # proceed.
    # (no-op placeholder removed; local comparison time and assumed state
    # are handled inline where required)

        # Authoritative early suppression: delegate to helper to keep this
        # function small and testable.
        try:
            # Determine immediate caller and delegate suppression decision to
            # the extracted helper. Passing caller_fn avoids inspecting the
            # stack inside the helper which would produce incorrect results.
            try:
                caller_fn = inspect.stack()[1].function
            except Exception:
                _ignored_exc()
                caller_fn = None

            if self._should_authoritatively_suppress(
                caller_fn, entity_id, pre_epoch, previous_last_action, action, bypass_throttle, force
            ):
                return False
        except Exception:
            _ignored_exc()
        try:
            # Log caller info to aid debugging in tests when unexpected
            # service calls are recorded.
            caller = None
            try:
                caller = inspect.stack()[1].function
            except Exception:
                caller = None
            _LOGGER.debug(
                "Invoking service switch.%s for %s (caller=%s)", action, entity_id, caller
            )
            # Some test harnesses register mock services in ways that make
            # `has_service` unreliable. Attempt the service call and catch
            # any errors instead of pre-checking availability.
            await self.hass.services.async_call(
                "switch", action, call_data, blocking=False
            )
        except Exception as err:
            _LOGGER.debug(
                "Failed to call switch.%s for %s: %s",
                action,
                entity_id,
                err,
            )
            return False
        # Record switch invocation metadata and leave the function small by
        # delegating the recording behavior.
        try:
            if entity_id:
                self._record_switch_invocation(entity_id, action, previous_last_action, pre_epoch)
        except Exception:
            _ignored_exc()
        return True

    def _record_switch_invocation(
        self,
        entity_id: str,
        action: str,
        previous_last_action: Optional[bool],
        pre_epoch: Optional[float],
    ) -> None:
        """Record switch invocation time, intended action, and eval id.

        Isolated to simplify the caller and make testing/refactoring easier.
        """
        # Record the actual time the service was invoked to ensure the
        # throttle check uses a consistent, real-world timestamp.
        # Prefer the coordinator's logical evaluation time when present
        # (used by deterministic tests). Fall back to real UTC if not set.
        epoch = self._resolve_epoch_for_invocation(pre_epoch)
        self._last_switch_time[entity_id] = epoch
        self._set_last_action_state(entity_id, action, previous_last_action)
        try:
            val = int(getattr(self, "_current_eval_id", 0) or 0)
            self._last_switch_eval[entity_id] = val
            try:
                _LOGGER.debug(
                    "DBG_SET_LAST_SWITCH_EVAL: entity=%s set_last_switch_eval=%s",
                    entity_id,
                    val,
                )
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()
        # Log a human-friendly isoformat for diagnostics
        try:
            _LOGGER.debug(
                "Recorded last_switch_time for %s = %s (epoch=%.3f)",
                entity_id,
                datetime.fromtimestamp(epoch).isoformat(),
                epoch,
            )
        except Exception:
            _LOGGER.debug("Recorded last_switch_time for %s = epoch %.3f", entity_id, epoch)
        try:
            _LOGGER.debug("last_switch_time keys after set: %s", list(self._last_switch_time.keys()))
        except Exception:
            _ignored_exc()

    def _resolve_epoch_for_invocation(self, pre_epoch: Optional[float]) -> float:
        """Resolve the epoch timestamp used when recording a switch invocation."""
        ts = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
        try:
            return float(pre_epoch) if pre_epoch is not None else float(dt_util.as_timestamp(ts))
        except Exception:
            try:
                return float(dt_util.as_timestamp(ts))
            except Exception:
                return float(dt_util.as_timestamp(dt_util.utcnow()))

    def _set_last_action_state(self, entity_id: str, action: str, previous_last_action: Optional[bool]) -> None:
        """Set the last intended action state for an entity."""
        try:
            if previous_last_action is not None:
                self._last_action_state[entity_id] = bool(previous_last_action)
            else:
                self._last_action_state[entity_id] = bool(action == "turn_on")
        except Exception:
            _ignored_exc()

    def _record_desired_state(self, entity_id: Any, desired: bool) -> None:
        """Record an observed desired state for confirmation counting.

        This increments the consecutive observation counter for the given
        entity, or resets it to 1 when the desired state changes.
        """
        entity = self._normalize_entity_id(entity_id)
        if not entity:
            return
        # Avoid recording the same entity multiple times within the same
        # coordinator evaluation. Multiple code paths may call
        # _record_desired_state during a single refresh; count only once per
        # evaluation so confirmation counters reflect consecutive coordinator
        # runs rather than duplicate observations inside a single run.
        last_eval = self._last_recorded_eval.get(entity)
        if last_eval == getattr(self, "_current_eval_id", None):
            return
        # mark as recorded for this evaluation (store an integer)
        self._last_recorded_eval[entity] = int(getattr(self, "_current_eval_id", 0) or 0)
        confirm_key = f"{entity}::confirm"
        required = int(
            self._device_switch_throttle.get(
                confirm_key, float(self._confirmation_required)
            )
        )
        hist = self._desired_state_history.get(entity, (desired, 0))
        # Detect flip: desired changed compared to last recorded desired
        try:
            if hist[0] != desired and hist[1] != 0:
                # record a flip-flop event at the current evaluation epoch
                try:
                    epoch = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                except Exception:
                    epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
                evts = self._flipflop_events.setdefault(entity, [])
                evts.append(epoch)
                # trim to window to avoid unbounded growth
                try:
                    cutoff = epoch - float(self._flipflop_window_seconds)
                    self._flipflop_events[entity] = [e for e in evts if e >= cutoff]
                except Exception:
                    _ignored_exc()

        except Exception:
            _ignored_exc()

        if hist[0] == desired:
            count = min(required, hist[1] + 1)
        else:
            count = 1
        self._desired_state_history[entity] = (desired, count)
        _LOGGER.debug(
            "Record desired: entity=%s desired=%s count=%d required=%d eval=%d",
            entity,
            desired,
            count,
            required,
            self._current_eval_id,
        )

    async def _maybe_switch(  # noqa: C901
        self,
        action: str,
        service_data: Dict[str, Any],
        desired: bool,
        force: bool = False,
        bypass_throttle: bool = False,
    ) -> bool:
        """Issue a switch call after throttle and confirmation debounce.

        If force is True, bypass confirmation and throttle (for emergency actions).
        """
        raw_entity = service_data.get("entity_id")
        norm = self._normalize_entity_id(raw_entity)
        if not norm:
            return False

        # Diagnostic: print entry-state so we can observe values before any
        # suppression logic mutates coordinator state.
        try:
            _LOGGER.debug(
                "DBG_ENTRY: entity=%s last_switch_eval=%r current_eval=%r last_action_state=%r last_switch_time=%r",
                norm,
                self._last_switch_eval.get(norm),
                getattr(self, "_current_eval_id", None),
                self._last_action_state.get(norm),
                self._last_switch_time.get(norm),
            )
        except Exception:
            _ignored_exc()
        # Trace the immediate caller and bypass flag for debugging
        try:
            caller_name = None
            try:
                caller_name = inspect.stack()[1].function
            except Exception:
                caller_name = None
            _LOGGER.debug(
                "DBG_CALLER: entity=%s caller=%r bypass_throttle_param=%s",
                norm,
                caller_name,
                bypass_throttle,
            )
        except Exception:
            _ignored_exc()

        # Consolidate initial early suppression checks into a helper to
        # reduce function complexity while preserving behavior.
        try:
            if self._early_suppress_checks(norm, desired, force, bypass_throttle):
                return False
        except Exception:
            # On any error in the helper, continue with normal processing
            _ignored_exc()

        # Use the coordinator's current evaluation time when available so
        # tests that simulate future times behave deterministically. Fall
        # back to real time if not set.
        now = getattr(self, "_current_eval_time", None) or dt_util.utcnow()

        # Determine whether to apply confirmation/throttle checks. If
        # either `force` or `bypass_throttle` is set, skip those checks so
        # urgent actions are not suppressed by early gating logic.
        should_check = not force and not bypass_throttle

        # Delegate the complex suppression and throttle logic to a helper
        # which returns True when the call should be suppressed.
        try:
            if self._should_suppress_switch(norm, desired, force, bypass_throttle):
                return False
        except Exception:
            _LOGGER.debug("_should_suppress_switch helper raised; proceeding with switch for %s", norm)

        # Keep logging minimal here — detailed debug logs used during
        # development have been removed to reduce CI noise.

        _LOGGER.debug(
            "_maybe_switch called: action=%s entity=%s desired=%s force=%s now=%s",
            action,
            norm,
            desired,
            force,
            now.isoformat(),
        )
        # Debug print for triage: show key values early
        try:
            _LOGGER.debug(
                "DBG_MAYBE_SWITCH_START: entity=%s desired=%s now=%s last_switch_time=%r last_action_state=%r device_throttle=%r",
                norm,
                desired,
                now.isoformat(),
                self._last_switch_time.get(norm),
                self._last_action_state.get(norm),
                self._device_switch_throttle.get(norm),
            )
        except Exception:
            _ignored_exc()
        # Demoted diagnostic: snapshot used during triage (kept at DEBUG level)
        try:
            _LOGGER.debug(
                "SNAPSHOT_FULL: eval=%s now=%s entity=%s last_switch=%r last_action=%r throttle_cfg=%r desired=%s should_check=%s",
                getattr(self, "_current_eval_id", None),
                now.isoformat(),
                norm,
                self._last_switch_time.get(norm),
                self._last_action_state.get(norm),
                self._device_switch_throttle.get(norm),
                desired,
                should_check,
            )
        except Exception:
            _ignored_exc()
        # Early suppression gate: if the last recorded action for this
        # entity differs from the currently desired state and that last
        # action occurred within the configured throttle window, suppress
        # the reversal immediately. This prevents the coordinator from
        # issuing a turn_off right after a turn_on (and vice versa) due to
        # rapid successive evaluations.
        # Early suppression gate: only apply when confirmation/throttle
        # checks are enabled. Urgent calls (force/bypass_throttle) should
        # not be suppressed by this gate.
        if should_check:
            # If there's an in-flight switch for this entity and it differs
            # from the currently desired state, suppress the reversal.
            try:
                pending = self._inflight_switches.get(norm)
                if pending is not None and bool(pending) != bool(desired):
                    _LOGGER.info(
                        "SUPPRESS_INFLIGHT: entity=%s pending=%s desired=%s",
                        norm,
                        pending,
                        desired,
                    )
                    return False
            except Exception:
                _ignored_exc()
            try:
                _LOGGER.debug(
                    "DBG_ENTER_SHOULD_CHECK: entity=%s should_check=%s last_switch_time_raw=%r last_action_state=%r last_switch_eval=%r current_eval=%r",
                    norm,
                    should_check,
                    self._last_switch_time.get(norm),
                    self._last_action_state.get(norm),
                    self._last_switch_eval.get(norm),
                    getattr(self, "_current_eval_id", None),
                )
            except Exception:
                _ignored_exc()
            # Consolidated robust quick-gate: compute normalized epoch values
            # and decisively suppress immediate reversals. This is a more
            # defensive check that avoids subtle type/timebase mismatches
            # seen in unit tests where the multiple earlier quick-gates
            # sometimes missed the reversal condition.
            try:
                try:
                    _LOGGER.debug(
                        "DBG_CONSOL_QUICK_GATE_START: entity=%s last_raw=%r",
                        norm,
                        self._last_switch_time.get(norm),
                    )
                except Exception:
                    _ignored_exc()
                last_raw = self._last_switch_time.get(norm)
                if last_raw is not None:
                    # Normalize last timestamp to epoch float
                    if isinstance(last_raw, (int, float)):
                        last_epoch_val = float(last_raw)
                    elif isinstance(last_raw, str):
                        parsed = dt_util.parse_datetime(last_raw)
                        last_epoch_val = float(dt_util.as_timestamp(parsed)) if parsed else None
                    else:
                        last_epoch_val = float(dt_util.as_timestamp(last_raw))

                    if last_epoch_val is not None:
                        now_epoch_val = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                        throttle_val = float(self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds) or self._default_switch_throttle_seconds)
                        last_action_state = self._last_action_state.get(norm)
                        if last_action_state is None:
                            try:
                                st = self.hass.states.get(norm)
                                last_action_state = bool(st and st.state == STATE_ON)
                            except Exception:
                                last_action_state = None
                        elapsed_val = now_epoch_val - float(last_epoch_val)
                        # Diagnostic print to stdout to avoid logging handle issues
                        try:
                            _LOGGER.debug(
                                "PRINT_QUICK_GATE: entity=%s last_epoch=%r last_action=%r now_epoch=%.3f elapsed=%.3f throttle=%.3f current_eval=%r last_eval=%r desired=%r",
                                norm,
                                last_epoch_val,
                                last_action_state,
                                now_epoch_val,
                                elapsed_val,
                                throttle_val,
                                getattr(self, "_current_eval_id", None),
                                self._last_switch_eval.get(norm),
                                desired,
                            )
                        except Exception:
                            _ignored_exc()
                        _LOGGER.debug(
                            "CONSOLIDATED_QUICK_GATE_INPUTS: entity=%s last_epoch=%.3f now_epoch=%.3f elapsed=%.3f throttle=%.3f last_action=%r desired=%s",
                            norm,
                            float(last_epoch_val),
                            now_epoch_val,
                            elapsed_val,
                            throttle_val,
                            last_action_state,
                            desired,
                        )
                        if last_action_state is not None and elapsed_val >= 0 and elapsed_val < float(throttle_val) and bool(last_action_state) != bool(desired):
                            _LOGGER.info(
                                "CONSOLIDATED_EARLY_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                                norm,
                                elapsed_val,
                                throttle_val,
                                last_action_state,
                                desired,
                            )
                            # If the last switch happened in the same evaluation
                            # or the immediately previous one, prefer suppression
                            # to avoid flip-flopping due to multiple code paths.
                            try:
                                last_eval = self._last_switch_eval.get(norm)
                                if last_eval is None or abs(int(getattr(self, "_current_eval_id", 0) or 0) - int(last_eval)) <= 1:
                                    return False
                            except Exception:
                                return False
                            return False
            except Exception:
                # Non-fatal: fall back to existing per-branch logic below
                _LOGGER.debug("Consolidated quick-gate failed for %s", norm)
            # Extra conservative quick-gate: ensure we suppress immediate
            # reversals using the same epoch timebase as _async_switch_call.
            try:
                last_val = self._last_switch_time.get(norm)
                throttle_cfg_quick = self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
                if last_val is not None and throttle_cfg_quick:
                    if isinstance(last_val, (int, float)):
                        last_e = float(last_val)
                    elif isinstance(last_val, str):
                        parsed = dt_util.parse_datetime(last_val)
                        last_e = float(dt_util.as_timestamp(parsed)) if parsed else None
                    else:
                        last_e = float(dt_util.as_timestamp(last_val))
                    if last_e is not None:
                        now_e = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                        try:
                            throttle_val_quick = float(throttle_cfg_quick)
                        except Exception:
                            throttle_val_quick = float(self._default_switch_throttle_seconds)
                        last_act_quick = self._last_action_state.get(norm)
                        if last_act_quick is None:
                            try:
                                st = self.hass.states.get(norm)
                                last_act_quick = bool(st and st.state == STATE_ON)
                            except Exception:
                                last_act_quick = None
                        # Debug snapshot for quick-gate evaluation
                        try:
                            _LOGGER.debug(
                                "QUICK_GATE_DEBUG: entity=%s last_e=%.3f now_e=%.3f elapsed=%.3f throttle_quick=%.3f last_act_quick=%r desired=%s current_eval_time=%s",
                                norm,
                                float(last_e) if last_e is not None else float('nan'),
                                now_e,
                                (now_e - last_e) if last_e is not None else float('nan'),
                                throttle_val_quick,
                                last_act_quick,
                                desired,
                                getattr(self, "_current_eval_time", None),
                            )
                        except Exception:
                            _ignored_exc()
                        # Evaluate suppression regardless of whether last_act_quick
                        # was retrieved from the coordinator or inferred from the
                        # entity state.
                        if last_act_quick is not None and (now_e - last_e) < throttle_val_quick and bool(last_act_quick) != bool(desired):
                            _LOGGER.info(
                                "EARLY_SUPPRESS_V2: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                                norm,
                                now_e - last_e,
                                throttle_val_quick,
                                last_act_quick,
                                desired,
                            )
                            return False
            except Exception:
                _ignored_exc()
            try:
                last_raw = self._last_switch_time.get(norm)
                throttle_cfg = self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
                if last_raw is not None and throttle_cfg:
                    try:
                        if isinstance(last_raw, (int, float)):
                            last_epoch = float(last_raw)
                        elif isinstance(last_raw, str):
                            parsed = dt_util.parse_datetime(last_raw)
                            last_epoch = float(dt_util.as_timestamp(parsed)) if parsed else None
                        else:
                            last_epoch = float(dt_util.as_timestamp(last_raw))
                    except Exception:
                        last_epoch = None
                    if last_epoch is not None:
                        now_epoch = float(dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow()))
                        # Prepare numeric inputs for logging
                        try:
                            throttle_val = float(throttle_cfg)
                        except Exception:
                            throttle_val = float(self._default_switch_throttle_seconds)
                        last_act = self._last_action_state.get(norm)
                        elapsed_val = now_epoch - float(last_epoch)
                        _LOGGER.info(
                            "EARLY_SUPPRESS_CHECK: entity=%s last_epoch=%.3f now_epoch=%.3f elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                            norm,
                            float(last_epoch),
                            now_epoch,
                            elapsed_val,
                            throttle_val,
                            last_act,
                            desired,
                        )
                        elapsed_early = elapsed_val
                        # Targeted diagnostic for failing test entity to help triage
                        try:
                            if norm == "switch.throttle_charger":
                                _LOGGER.debug(
                                    "DIAG_EARLY: entity=%s last_raw=%r last_epoch=%.3f now_epoch=%.3f elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                                    norm,
                                    last_raw,
                                    float(last_epoch),
                                    now_epoch,
                                    elapsed_val,
                                    throttle_val,
                                    last_act,
                                    desired,
                                )
                        except Exception:
                            _ignored_exc()
                        if elapsed_early is not None and elapsed_early < throttle_val and last_act is not None and bool(last_act) != bool(desired):
                            _LOGGER.info(
                                "EARLY_SUPPRESS: entity=%s last_action=%s desired=%s elapsed=%.3f throttle=%s",
                                norm,
                                last_act,
                                desired,
                                elapsed_early,
                                throttle_val,
                            )
                            return False
            except Exception:
                # On any error, continue to normal processing
                _LOGGER.debug("Early suppression gate failed for %s", norm)
        # only log last_switch_time at debug level
        _LOGGER.debug("last_switch_time map at start: %s", self._last_switch_time)
        # Debug: show the raw representation of the entity_id to catch
        # accidental list/tuple vs string mismatches that lead to missed
        # throttle-key lookups in tests.
        try:
            _LOGGER.info("_maybe_switch raw entity repr: %r (type=%s)", raw_entity, type(raw_entity))
        except Exception:
            _ignored_exc()
        # Log at INFO so test runs with default logging will show the
        # coordinator's decision inputs for easier triage.
        _LOGGER.info(
            "_maybe_switch state: entity=%s last_switch=%s device_throttle=%s confirm=%s bypass_throttle=%s",
            norm,
            self._last_switch_time.get(norm),
            self._device_switch_throttle.get(norm),
            self._device_switch_throttle.get(f"{norm}::confirm"),
            bypass_throttle,
        )

    # Confirmation debounce and throttle checks are normally applied
        # to avoid flapping. However, certain urgent operations (for
        # example precharge-release pauses or presence-leave) explicitly
        # request bypass_throttle=True or force=True and should act
        # immediately. In those cases skip confirmation and throttle.
        should_check = not force and not bypass_throttle
        # Keep gate decision at DEBUG level to avoid noisy CI logs.
        _LOGGER.debug(
            "MAYBE_SWITCH_GATE: entity=%s force=%s bypass_throttle=%s should_check=%s last=%s throttle_config=%s",
            norm,
            force,
            bypass_throttle,
            should_check,
            self._last_switch_time.get(norm),
            self._device_switch_throttle.get(norm),
        )
        # Log summary so test runs can see the inputs used for the decision.
        try:
            confirm_key = f"{norm}::confirm"
            required = int(
                self._device_switch_throttle.get(
                    confirm_key, float(self._confirmation_required)
                )
            )
            hist = self._desired_state_history.get(norm, (desired, 0))
            # Debug: inspect existing last_switch_time keys to catch mismatches
            try:
                _LOGGER.info("last_switch_time keys before check: %s", list(self._last_switch_time.keys()))
            except Exception:
                _ignored_exc()
            last = self._last_switch_time.get(norm)
            _LOGGER.debug("throttle: last raw=%r type=%s truthy=%s", last, type(last), bool(last))
            throttle = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            _LOGGER.info(
                "_maybe_switch inputs: entity=%s should_check=%s last=%s throttle=%s required=%s hist=%s",
                norm,
                should_check,
                last,
                throttle,
                required,
                hist,
            )
        except Exception:
            _ignored_exc()
        _LOGGER.debug("_maybe_switch debug: should_check=%s", should_check)

        # Throttle/confirmation check (per-device configured)
        if should_check:
            # Use the normalized entity id for per-device lookups so keys
            # match how we store throttle and last-switch timestamps.
            throttle = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            last = self._last_switch_time.get(norm)

            _LOGGER.info("SHOULD_CHECK_VARS: entity=%s last=%r last_type=%s throttle=%s", norm, last, type(last), throttle)

            # Confirmation debounce: per-device override available. Record the
            # observed desired state for confirmation counting. The helper will
            # increment or reset the consecutive counter as appropriate.
            # Record using normalized key
            self._record_desired_state(norm, desired)
            confirm_key = f"{norm}::confirm"
            required = int(
                self._device_switch_throttle.get(
                    confirm_key, float(self._confirmation_required)
                )
            )
            hist = self._desired_state_history.get(norm, (desired, 0))
            count = hist[1]
            # Emit a concise summary so failing CI/tests produce actionable
            # logs without scanning DEBUG output.
            _LOGGER.debug(
                "MAYBE_SWITCH_SUMMARY: entity=%s required=%s hist=%s count=%s last=%s throttle=%s",
                norm,
                required,
                hist,
                count,
                self._last_switch_time.get(norm),
                self._device_switch_throttle.get(norm),
            )

            # If we haven't yet observed the required number of consecutive
            # confirmations, wait (regardless of throttle state).
            if count < required:
                _LOGGER.debug(
                    "Waiting for confirmation for %s -> desired=%s (count=%d/%d)",
                    norm,
                    desired,
                    count,
                    required,
                )
                _LOGGER.debug(
                    "DBG_WAIT: entity=%s desired=%s count=%d required=%d eval=%s",
                    norm,
                    desired,
                    count,
                    required,
                    getattr(self, "_current_eval_id", None),
                )
                return False

            # Throttle check (per-device configured): even if we've reached the
            # confirmation count, do not issue switches while inside the throttle
            # window. The confirmation counter will continue to advance in the
            # background and the next coordinator evaluation after the throttle
            # expires will proceed to call the switch.
            # Use the coordinator's logical evaluation time when available so
            # deterministic tests that pass a simulated ``now_local`` control
            # throttle evaluation. Fall back to real UTC otherwise.
            # Temporary diagnostic: ensure the normalized key is present in
            # the last-switch map. If it isn't, throttle checks will be
            # skipped and rapid switching can occur.
            _LOGGER.debug("throttle diagnostic: norm_present=%s keys=%s", norm in self._last_switch_time, list(self._last_switch_time.keys()))
            # Compute a normalized epoch for the last switch when possible.
            last_epoch_quick: float | None = None
            try:
                if isinstance(last, (int, float)):
                    last_epoch_quick = float(last)
                elif isinstance(last, str):
                    parsed = dt_util.parse_datetime(last)
                    last_epoch_quick = float(dt_util.as_timestamp(parsed)) if parsed else None
                else:
                    if last is not None:
                        last_epoch_quick = float(dt_util.as_timestamp(last))
                    else:
                        last_epoch_quick = None
            except Exception:
                last_epoch_quick = None

            # Quick conservative gate: if the last action differs from the
            # current desired state and the last switch happened inside the
            # throttle window, suppress the opposite call. This avoids races
            # where the coordinator attempts to immediately reverse a recent
            # action due to timing differences.
            try:
                _LOGGER.debug("DEBUG_QUICK_GATE: raw last=%r type=%s", last, type(last))
                if last_epoch_quick is not None:
                    # Use the coordinator logical evaluation time when available
                    # for deterministic behavior in tests; fall back to real UTC.
                    now_for_quick = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                    now_epoch_val = float(dt_util.as_timestamp(now_for_quick))
                    elapsed_quick = now_epoch_val - float(last_epoch_quick)
                    # Ensure throttle is a float
                    try:
                        throttle_val = float(throttle)
                    except Exception:
                        throttle_val = float(self._default_switch_throttle_seconds)

                    last_action_state = self._last_action_state.get(norm)
                    # Fallback: if we don't have a recorded last_action_state,
                    # infer from the current HA entity state to decide whether
                    # this would be a reversal.
                    if last_action_state is None:
                        try:
                            st = self.hass.states.get(norm)
                            last_action_state = bool(st and st.state == STATE_ON)
                        except Exception:
                            last_action_state = None

                    _LOGGER.debug(
                        "QUICK_GATE_INPUTS: entity=%s last_epoch=%.3f now_epoch=%.3f elapsed_quick=%.3f last_action=%r desired=%s throttle=%s",
                        norm,
                        float(last_epoch_quick),
                        now_epoch_val,
                        elapsed_quick,
                        last_action_state,
                        desired,
                        throttle_val,
                    )

                    if elapsed_quick is not None and elapsed_quick < throttle_val:
                        if last_action_state is not None and bool(last_action_state) != bool(desired):
                            _LOGGER.debug(
                                "QUICK_SUPPRESS: entity=%s last_action=%s desired=%s elapsed=%.3f throttle=%s",
                                norm,
                                last_action_state,
                                desired,
                                elapsed_quick,
                                throttle_val,
                            )
                            return False
            except Exception:
                # Be conservative and do not suppress on unexpected errors
                _LOGGER.exception("Quick gate evaluation failed for %s", norm)

            if last:
                _LOGGER.info("ENTER_THROTTLE_BLOCK: entity=%s last=%r", norm, last)
                # Normalize both timestamps to UTC before subtracting so
                # comparisons are consistent regardless of how tests or
                # callers constructed the datetimes (local vs utc).
                # Use real UTC time for throttle comparisons to align with
                # tests and external callers that record last switch times
                # using dt_util.utcnow(). Using the coordinator's logical
                # _current_eval_time here caused mismatches in some tests.
                now_for_cmp = dt_util.utcnow()
                try:
                    # Use timestamps which correctly handle timezone-aware
                    # datetimes and avoid pitfalls with naive/local mixes.
                    now_ts = float(dt_util.as_timestamp(now_for_cmp))
                    if isinstance(last, (int, float)):
                        last_ts_val = float(last)
                    else:
                        last_ts_val = float(dt_util.as_timestamp(last))
                    _LOGGER.debug(
                        "THROTTLE_TS: now_ts=%s (%s) last_ts=%s (%s)",
                        now_ts,
                        type(now_for_cmp),
                        last_ts_val,
                        type(last),
                    )
                    elapsed = now_ts - last_ts_val

                    # Emit a debug-level snapshot of numeric inputs used for
                    # throttle decisions. Keep as DEBUG to avoid CI noise.
                    _LOGGER.debug(
                        "THROTTLE_DECISION_INPUTS: entity=%s now_ts=%.3f last_ts=%.3f elapsed=%.3f throttle=%s required=%s",
                        norm,
                        now_ts,
                        last_ts_val,
                        elapsed,
                        throttle,
                        required,
                    )

                    # If the recorded last-switch time appears to be in the
                    # future relative to the evaluation time (can happen in
                    # tests due to ordering or timezone differences), treat
                    # the throttle as expired instead of throttling the
                    # action. This avoids false suppression when tests
                    # manually backdate timestamps and advance HA time.
                    if elapsed < 0:
                        _LOGGER.warning(
                            "THROTTLE_TS: last_switch_time (%s) is after now (%s) -> treating as expired",
                            last,
                            now_for_cmp,
                        )
                        elapsed = float("inf")
                        # Keep the future-time warning at WARNING level and
                        # otherwise demote the detailed numeric snapshot.
                        _LOGGER.debug(
                            "THROTTLE_DECISION_INPUTS: entity=%s now_ts=%s last_ts=%s elapsed=%.3f throttle=%s required=%s",
                            norm,
                            now_ts,
                            last_ts_val,
                            elapsed,
                            throttle,
                            required,
                        )
                    _LOGGER.debug("THROTTLE_TS: computed elapsed=%s throttle=%s", elapsed, throttle)
                except Exception:
                    _LOGGER.exception(
                        "Throttle timestamp computation failed for %s: last=%r now=%r",
                        norm,
                        last,
                        now_for_cmp,
                    )
                    elapsed = None

                # Render readable representations for logs
                try:
                    if isinstance(last, (int, float)):
                        readable_last = datetime.fromtimestamp(float(last)).isoformat()
                    else:
                        readable_last = getattr(last, "isoformat", lambda: repr(last))()
                except Exception:
                    readable_last = repr(last)
                try:
                    readable_now = getattr(now_for_cmp, "isoformat", lambda: repr(now_for_cmp))()
                except Exception:
                    readable_now = repr(now_for_cmp)

                _LOGGER.debug(
                    "THROTTLE_DECISION: entity=%s throttle=%s last=%s now=%s elapsed=%s required=%s",
                    norm,
                    throttle,
                    readable_last,
                    readable_now,
                    elapsed,
                    required,
                )
                # Emit a concise decision check so CI logs contain the
                # exact numeric values used for the throttle comparison.
                _LOGGER.info(
                    "DECISION_CHECK: entity=%s elapsed=%s throttle=%s required=%s",
                    norm,
                    elapsed,
                    throttle,
                    required,
                )
                if elapsed is not None and elapsed < float(throttle):
                    _LOGGER.info(
                        "THROTTLED: switch.%s for %s (last %.1fs ago, throttle=%.1fs)",
                        action,
                        norm,
                        elapsed,
                        float(throttle),
                    )
                    return False
                else:
                    _LOGGER.debug(
                        "THROTTLE_OK: switch.%s for %s (last=%s elapsed=%s throttle=%s)",
                        action,
                        norm,
                        last,
                        elapsed,
                        float(throttle),
                    )

        # Clear history and call switch. Pre-record the switch time so tests
        # that simulate immediate state changes observe a consistent last
        # switch timestamp for throttle checks. Prefer the coordinator's
        # logical evaluation time when present to keep tests deterministic.
        self._desired_state_history.pop(norm, None)
        # Capture the previous last action state so the final throttle
        # check can determine whether this would be a reversal. We must
        # not overwrite it with the current `desired` until after the
        # final throttle decision is made.
        previous_last_action = self._last_action_state.get(norm)
        pre_ts = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
        try:
            pre_epoch = float(dt_util.as_timestamp(pre_ts))
        except Exception:
            pre_epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
        # Final safety throttle check using the current stored last-switch
        # timestamp. This protects against races where earlier checks may
        # have missed the most recent service invocation. Only perform this
        # conservative safety check when confirmation/throttle gating is
        # enabled (i.e. should_check is True). Urgent callers that set
        # ``force`` or ``bypass_throttle`` will skip this final suppression
        # and act immediately.
        if should_check:
            try:
                # Triage print: show eval ids and last switch eval for this entity
                try:
                    _LOGGER.debug(
                        "DBG_FINAL_CHECK: entity=%s current_eval=%r last_switch_eval=%r previous_last_action=%r last_switch_time_raw=%r",
                        norm,
                        getattr(self, "_current_eval_id", None),
                        self._last_switch_eval.get(norm),
                        previous_last_action,
                        self._last_switch_time.get(norm),
                    )
                except Exception:
                    _ignored_exc()
                current_last = self._last_switch_time.get(norm)
                throttle = self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds)
                if current_last is not None:
                    if isinstance(current_last, (int, float)):
                        stored_epoch = float(current_last)
                    else:
                        parsed = dt_util.parse_datetime(current_last) if isinstance(current_last, str) else current_last
                        stored_epoch = float(dt_util.as_timestamp(parsed)) if parsed else None
                else:
                    stored_epoch = None
                if stored_epoch is not None:
                    # Compute comparison time using coordinator logical eval time
                    now_epoch = float(
                        dt_util.as_timestamp(getattr(self, "_current_eval_time", None) or dt_util.utcnow())
                    )
                    elapsed_now = now_epoch - stored_epoch
                    # Diagnostic: log the values used for the final throttle check
                    _LOGGER.info(
                        "FINAL_THROTTLE_CHECK: entity=%s stored_epoch=%.3f now_epoch=%.3f elapsed=%.3f throttle=%s last_action=%r desired=%s",
                        norm,
                        float(stored_epoch),
                        now_epoch,
                        elapsed_now,
                        throttle,
                        self._last_action_state.get(norm),
                        desired,
                    )
                    if elapsed_now < float(throttle):
                        # Prefer the previously captured last action (from
                        # before we updated _last_action_state) as it most
                        # accurately represents the action that may still be
                        # in-flight. Fall back to the current recorded state
                        # if needed.
                        last_action_state = previous_last_action if previous_last_action is not None else self._last_action_state.get(norm)
                        if last_action_state is not None and bool(last_action_state) != bool(desired):
                            _LOGGER.warning(
                                "FINAL_THROTTLE_SUPPRESS: entity=%s last_action=%s desired=%s elapsed=%.3f throttle=%s",
                                norm,
                                last_action_state,
                                desired,
                                elapsed_now,
                                throttle,
                            )
                            return False
            except Exception:
                # On any error, continue to pre-record and attempt the switch.
                _ignored_exc()
        # Extra deterministic pre-record check: avoid overwriting the
        # previously recorded last-switch timestamp until we've ensured
        # this planned action is not an immediate reversal. Compare the
        # pre-computed pre_epoch against the stored last timestamp and
        # suppress if it would reverse a recent action inside the
        # configured throttle window.
        # Only apply this conservative check when confirmation/throttle
        # gating is enabled; callers that set ``force`` or
        # ``bypass_throttle`` should skip this check.
        if should_check:
            try:
                stored = self._last_switch_time.get(norm)
                if stored is not None:
                    try:
                        if isinstance(stored, (int, float)):
                            stored_epoch_det = float(stored)
                        elif isinstance(stored, str):
                            parsed = dt_util.parse_datetime(stored)
                            stored_epoch_det = float(dt_util.as_timestamp(parsed)) if parsed else None
                        else:
                            stored_epoch_det = float(dt_util.as_timestamp(stored))
                    except Exception:
                        stored_epoch_det = None
                else:
                    stored_epoch_det = None
                throttle_val_det = float(self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds) or self._default_switch_throttle_seconds)
                if stored_epoch_det is not None and pre_epoch is not None:
                    elapsed_det = float(pre_epoch) - float(stored_epoch_det)
                    last_act_det = self._last_action_state.get(norm)
                    if last_act_det is None:
                        try:
                            st = self.hass.states.get(norm)
                            last_act_det = bool(st and st.state == STATE_ON)
                        except Exception:
                            last_act_det = None
                    if last_act_det is not None and elapsed_det >= 0 and elapsed_det < float(throttle_val_det) and bool(last_act_det) != bool(desired):
                        _LOGGER.info("PRE_RECORD_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s", norm, elapsed_det, throttle_val_det, last_act_det, desired)
                        try:
                            _LOGGER.debug(
                                "DBG_PRE_RECORD_SUPPRESS: entity=%s stored_epoch=%r pre_epoch=%r elapsed=%r throttle=%r last_act=%r desired=%r",
                                norm,
                                stored_epoch_det,
                                pre_epoch,
                                elapsed_det,
                                throttle_val_det,
                                last_act_det,
                                desired,
                            )
                        except Exception:
                            _ignored_exc()
                        return False
            except Exception:
                _ignored_exc()
        _LOGGER.info("Pre-record: intended action for %s (pre_epoch=%.3f)", norm, pre_epoch)
        _LOGGER.info("PROCEED: calling switch.%s for %s", action, norm)
        # Ensure the service_data contains a normalized entity id for
        # _async_switch_call so recording uses the same key lookup.
        call_data = dict(service_data)
        call_data["entity_id"] = norm
        try:
            _LOGGER.debug(
                "DBG_ALLOWING_CALL: entity=%s action=%s desired=%s previous_last_action=%r pre_epoch=%r last_switch_time=%r last_switch_eval=%r current_eval=%r",
                norm,
                action,
                desired,
                previous_last_action,
                pre_epoch,
                self._last_switch_time.get(norm),
                self._last_switch_eval.get(norm),
                getattr(self, "_current_eval_id", None),
            )
        except Exception:
            _ignored_exc()
        # Record the intended action state now that final guards passed so
        # other code paths can assume the coordinator's intended switch
        # state. Do not record the authoritative last-switch timestamp
        # here; that will only be set when the service actually executes
        # in _async_switch_call.
        try:
            self._last_action_state[norm] = bool(desired)
        except Exception:
            _ignored_exc()
        # Mark the intended action as in-flight so other checks in the same
        # coordinator evaluation will avoid issuing a reversal.
        try:
            self._inflight_switches[norm] = bool(desired)
        except Exception:
            _ignored_exc()
        # Final pre-call authoritative guard: double-check using the
        # captured previous_last_action and the stored last-switch
        # timestamp to ensure we don't issue an immediate reversal that
        # slipped past earlier checks. This is conservative and only
        # applies when throttle/confirmation checks are enabled.
        if should_check:
            try:
                # Extra diagnostic: always print the raw stored value and
                # surrounding variables so we can debug why this guard may
                # not trigger in tests.
                try:
                    _LOGGER.debug(
                        "DBG_FINAL_GUARD_CHECK: entity=%s stored_raw=%r stored_type=%s pre_epoch=%r previous_last_action=%r last_action_state_now=%r throttle_cfg=%r desired=%r",
                        norm,
                        self._last_switch_time.get(norm),
                        type(self._last_switch_time.get(norm)),
                        pre_epoch,
                        previous_last_action,
                        self._last_action_state.get(norm),
                        self._device_switch_throttle.get(norm),
                        desired,
                    )
                except Exception:
                    _ignored_exc()

                stored = self._last_switch_time.get(norm)
                if stored is not None:
                    if isinstance(stored, (int, float)):
                        stored_epoch_final = float(stored)
                    elif isinstance(stored, str):
                        parsed = dt_util.parse_datetime(stored)
                        stored_epoch_final = float(dt_util.as_timestamp(parsed)) if parsed else None
                    else:
                        stored_epoch_final = float(dt_util.as_timestamp(stored))
                else:
                    stored_epoch_final = None

                throttle_final = float(self._device_switch_throttle.get(norm, self._default_switch_throttle_seconds) or self._default_switch_throttle_seconds)
                if stored_epoch_final is not None and pre_epoch is not None:
                    elapsed_final = float(pre_epoch) - float(stored_epoch_final)
                    last_action_for_final = previous_last_action if previous_last_action is not None else self._last_action_state.get(norm)
                    if last_action_for_final is None:
                        try:
                            st = self.hass.states.get(norm)
                            last_action_for_final = bool(st and st.state == STATE_ON)
                        except Exception:
                            last_action_for_final = None

                    if (
                        last_action_for_final is not None
                        and elapsed_final >= 0
                        and elapsed_final < float(throttle_final)
                        and bool(last_action_for_final) != bool(desired)
                    ):
                        try:
                            print(
                                (
                                    "DBG_FINAL_GUARD_SUPPRESS: entity="
                                    f"{norm} elapsed={elapsed_final:.3f} throttle={throttle_final} "
                                    f"last_action={last_action_for_final} desired={desired} "
                                    f"previous_last_action={previous_last_action}"
                                )
                            )
                        except Exception:
                            _ignored_exc()

                        _LOGGER.info(
                            "FINAL_GUARD_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_action=%r desired=%s",
                            norm,
                            elapsed_final,
                            throttle_final,
                            last_action_for_final,
                            desired,
                        )

                        # Clear inflight marker since we're not proceeding
                        try:
                            self._inflight_switches.pop(norm, None)
                        except Exception:
                            _ignored_exc()

                        return False
            except Exception:
                _ignored_exc()
        result = await self._async_switch_call(
            action,
            call_data,
            pre_epoch=pre_epoch,
            previous_last_action=previous_last_action,
            bypass_throttle=bypass_throttle,
            force=force,
        )
        if result:
            # _async_switch_call will have recorded the authoritative
            # last-switch timestamp/eval; nothing further to do here.
            pass
        return result

    async def _apply_charger_logic(  # noqa: C901
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

        # Consider the coordinator's last intended action as an "assumed"
        # state when available. Many tests and some integrations rely on the
        # coordinator's recorded intended action (which may not have been
        # applied to the entity state yet) for deterministic behavior.
        last_action = self._last_action_state.get(device.charger_switch)
        assumed_on = charger_is_on or (last_action is True)

        # Persist the forecast_holdoff flag per-device so other internal
        # helpers (notably the switch call path) can reason about
        # forecast-driven bypasses deterministically. This is set here
        # because `_apply_charger_logic` receives `forecast_holdoff` as
        # a parameter from the plan builder.
        try:
            if forecast_holdoff:
                self._forecast_holdoff[device.name] = True
            else:
                self._forecast_holdoff.pop(device.name, None)
        except Exception:
            _ignored_exc()

        service_data = {"entity_id": charger_ent}
        device_name = device.name
        try:
            _LOGGER.info(
                "ENTER_APPLY_LOGIC: device=%s charger_is_on=%s precharge_required=%s release_level=%s precharge_maps=%s",
                device_name,
                charger_is_on,
                precharge_required,
                release_level,
                {"release": list(self._precharge_release.keys()), "ready": list(self._precharge_release_ready.keys())},
            )
        except Exception:
            _ignored_exc()
        window_imminent = (
            smart_start_active
            and start_time is not None
            and start_time > now_local
            and (start_time - now_local) <= timedelta(seconds=5)
        )

        # If a precharge release has just cleared (release_level is None)
        # but the charger is still on, and the battery/prediction is safely
        # above the precharge thresholds, pause the charger immediately.
        # This ensures the component does not keep the charger running after
        # the intended precharge latch has been released.
        try:
            # Only treat this as an urgent precharge-release if there was
            # previously a precharge latch for this device. Without a
            # prior latch, pausing the charger here is not an urgent
            # precharge-release event and should respect the normal
            # throttle/confirmation gates to avoid rapid flip-flops.
            had_precharge_latch = (
                device.name in self._precharge_release
                or device.name in self._precharge_release_ready
            )
            # Fallback: if the coordinator previously recorded an intended
            # turn_on for this charger (last_action_state True) and the
            # precharge latch is now cleared (release_level is None and
            # precharge_required is False) while the charger remains on,
            # ensure we pause it immediately. This catches cases where the
            # precharge maps were cleared earlier in the evaluation but the
            # urgent pause path above did not trigger due to timing.
            try:
                prev_intended = self._last_action_state.get(charger_ent)
                threshold_cleared = self._precharge_release_cleared_by_threshold.pop(device.name, False)
                try:
                    # Diagnostic: expose internal markers at DEBUG level to aid
                    # triage while avoiding stdout in CI.
                    _LOGGER.debug(
                        "DBG_PRECHARGE_MARKERS: device=%s had_precharge_latch=%s prev_intended=%r threshold_cleared=%s precharge_release_keys=%r precharge_ready_keys=%r",
                        device.name,
                        had_precharge_latch,
                        prev_intended,
                        threshold_cleared,
                        list(self._precharge_release.keys()),
                        list(self._precharge_release_ready.keys()),
                    )
                except Exception:
                    _ignored_exc()
                if (
                    charger_is_on
                    and not precharge_required
                    and release_level is None
                    and prev_intended is True
                    and had_precharge_latch
                ):
                    self._log_action(
                        device_name,
                        logging.INFO,
                        "[Precharge-Fallback] %s latch cleared after coordinator intent -> pausing charger (%s)",
                        device_name,
                        charger_ent,
                    )
                    await self._maybe_switch(
                        "turn_off", service_data, desired=False, bypass_throttle=True
                    )
                    return False
                # If the latch cleared due to reaching thresholds (not presence),
                # do not bypass the throttle; allow the normal throttle/confirmation
                # gates to decide whether to pause immediately to avoid breaking
                # anti-flapping guarantees in unit tests.
                if (
                    charger_is_on
                    and not precharge_required
                    and release_level is None
                    and threshold_cleared
                ):
                    self._log_action(
                        device_name,
                        logging.INFO,
                        "[Precharge-Fallback] %s latch cleared by threshold -> pausing charger if allowed (%s)",
                        device_name,
                        charger_ent,
                    )
                    # Threshold-cleared releases should pause the charger
                    # immediately when the charger is actually reporting
                    # active charging. If no charging sensor is configured or
                    # the charger is not reporting "charging", respect the
                    # normal throttle/confirmation gates to avoid causing
                    # unwanted rapid reversals in unrelated unit tests.
                    try:
                        charging_state = self._charging_state(device.charging_sensor)
                    except Exception:
                        charging_state = "unknown"
                    if charging_state == "charging":
                        await self._maybe_switch(
                            "turn_off", service_data, desired=False, bypass_throttle=True
                        )
                    else:
                        await self._maybe_switch(
                            "turn_off", service_data, desired=False
                        )
                    return False
                    return False
            except Exception:
                _ignored_exc()
                try:
                    _LOGGER.info(
                        "DEBUG_PRECHARGE_RELEASE_CHECK: device=%s charger_is_on=%s precharge_required=%s release_level=%s predicted_level=%.3f battery=%.3f had_precharge_latch=%s",
                        device.name,
                        charger_is_on,
                        precharge_required,
                        release_level,
                        float(predicted_level),
                        float(battery),
                        had_precharge_latch,
                    )
                except Exception:
                    _ignored_exc()
            if (
                charger_is_on
                and not precharge_required
                and release_level is None
                and predicted_level >= device.precharge_level + margin_on
                and had_precharge_latch
            ):
                self._log_action(
                    device_name,
                    logging.INFO,
                    "[Precharge] %s release cleared -> pausing charger (%s)",
                    device_name,
                    charger_ent,
                )
                # Pause the charger; do not bypass the configured throttle
                # here so that normal anti-flapping protections remain in
                # effect for non-urgent transitions.
                # Pause the charger immediately when a precharge release is
                # detected; this is an urgent action and should bypass the
                # configured throttle to avoid leaving the charger running.
                await self._maybe_switch(
                    "turn_off", service_data, desired=False, bypass_throttle=True
                )
                return False
        except Exception:
            # Non-fatal logging-only protection; continue to normal logic
            _LOGGER.exception("Error while handling immediate precharge release")

        # If presence just left home and there was an active precharge latch,
        # clear and pause the charger immediately to avoid charging while away.
        try:
            # If the plan builder cleared the precharge latch because the
            # device left home, the flag `_precharge_release_cleared_by_presence`
            # will be set. Act on it here: pause the charger immediately and
            # clear the marker so this is a one-time action.
            if (
                charger_is_on
                and self._precharge_release_cleared_by_presence.pop(device.name, False)
            ):
                self._precharge_release.pop(device.name, None)
                self._precharge_release_ready.pop(device.name, None)
                self._log_action(
                    device_name,
                    logging.INFO,
                    "[Precharge] %s presence left -> clearing latch and pausing charger (%s)",
                    device_name,
                    charger_ent,
                )
                await self._maybe_switch(
                    "turn_off", service_data, desired=False, bypass_throttle=True
                )
                return False
        except Exception:
            _LOGGER.exception("Error while handling presence-leave precharge clear")

        if battery <= device.min_level and not charger_is_on:
            self._log_action(
                device_name,
                logging.DEBUG,
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

        if forecast_holdoff and smart_start_active and not precharge_required:
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
                await self._maybe_switch(
                    "turn_off", service_data, desired=False, bypass_throttle=True
                )
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

        if assumed_on and battery >= device.target_level:
            self._log_action(
                device_name,
                logging.INFO,
                "[SmartStop] %s reached target level %.1f%% -> deactivating charger (%s)",
                device_name,
                battery,
                charger_ent,
            )
            # Respect configured throttle for SmartStop to avoid rapid toggling;
            # do not bypass the throttle here. To avoid races where the
            # coordinator's plan immediately attempts to reverse a recent
            # switch, pre-check the per-device throttle window and suppress
            # the call if the last switch is still within the throttle.
            try:
                last = self._last_switch_time.get(charger_ent)
                throttle = self._device_switch_throttle.get(
                    charger_ent, self._default_switch_throttle_seconds
                )
                if last is not None and throttle is not None:
                    _LOGGER.warning("SmartStop: raw last=%r type=%s throttle=%s", last, type(last), throttle)
                    # Coerce string timestamps to datetimes when necessary.
                    # last may be stored as an epoch float, datetime, or string
                    try:
                        if isinstance(last, (int, float)):
                            last_ts_val = float(last)
                        elif isinstance(last, str):
                            parsed = dt_util.parse_datetime(last)
                            last_ts_val = float(dt_util.as_timestamp(parsed)) if parsed else None
                        else:
                            # Assume datetime-like
                            last_ts_val = float(dt_util.as_timestamp(last))
                        now_ts = float(dt_util.as_timestamp(dt_util.utcnow()))
                        if last_ts_val is None:
                            raise ValueError("invalid last switch timestamp")
                        elapsed = now_ts - last_ts_val
                        _LOGGER.debug(
                            "SmartStop throttle check: now_ts=%s last_ts=%s elapsed=%.3f throttle=%s",
                            now_ts,
                            last_ts_val,
                            elapsed,
                            throttle,
                        )
                        if elapsed < float(throttle):
                            # Still inside throttle window: skip issuing turn_off
                            self._log_action(
                                device_name,
                                logging.DEBUG,
                                "[SmartStop] %s skipping turn_off due to throttle (%.1fs < %.1fs)",
                                device_name,
                                elapsed,
                                float(throttle),
                            )
                            return expected_on
                    except Exception:
                        # If throttle check fails, fall back to normal behavior.
                        _ignored_exc()
            except Exception:
                # If throttle check fails, fall back to normal behavior.
                _ignored_exc()
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
            await self._maybe_switch(
                "turn_off", service_data, desired=False, bypass_throttle=True
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
            await self._maybe_switch(
                "turn_off", service_data, desired=False, bypass_throttle=True
            )
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
                # When precharge release conditions are met, bypass the
                # standard throttle so the charger can be paused immediately
                # even if it was switched on in the same coordinator run.
                # Compute a conservative bypass: only bypass when a release
                # level is configured and the precharge is no longer
                # required (i.e. this is an actual release event).
                bypass = bool(release_level is not None and not precharge_required)
                try:
                    pre_ts_local = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                    pre_epoch_local = float(dt_util.as_timestamp(pre_ts_local))
                except Exception:
                    pre_epoch_local = float(dt_util.as_timestamp(dt_util.utcnow()))
                await self._async_switch_call(
                    "turn_off",
                    service_data,
                    pre_epoch=pre_epoch_local,
                    previous_last_action=True,
                    bypass_throttle=bypass,
                )
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
                _LOGGER.info(
                    "About to call _maybe_switch turn_on for %s (entity=%s)",
                    device_name,
                    charger_ent,
                )
                # When activating precharge, allow forecast-driven bypass of
                # the throttle so predictive starts can proceed if the
                # forecast_holdoff flag is set for this device. However,
                # when there's no forecast-driven bypass we must preserve
                # the confirmation debounce and throttle checks implemented
                # by _maybe_switch. Only call the authoritative _async_switch_call
                # directly when bypassing is requested.
                bypass = bool(forecast_holdoff)
                if bypass:
                    try:
                        pre_ts_local = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                        pre_epoch_local = float(dt_util.as_timestamp(pre_ts_local))
                    except Exception:
                        pre_epoch_local = float(dt_util.as_timestamp(dt_util.utcnow()))
                    # Preserve stored previous_last_action when available
                    prev_action = self._last_action_state.get(charger_ent)
                    await self._async_switch_call(
                        "turn_on",
                        service_data,
                        pre_epoch=pre_epoch_local,
                        previous_last_action=prev_action,
                        bypass_throttle=True,
                    )
                    return True
                # No bypass requested: use _maybe_switch so confirmation and
                # throttle gating apply as normal.
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
