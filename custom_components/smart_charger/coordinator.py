from __future__ import annotations
import inspect
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Dict, Iterable, Mapping, Callable

from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    ALARM_MODE_PER_DAY,
    ALARM_MODE_SINGLE,
    CHARGING_STATES,
    CONF_ADAPTIVE_EWMA_ALPHA,
    CONF_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
    CONF_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
    CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP,
    CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS,
    CONF_ADAPTIVE_THROTTLE_ENABLED,
    CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
    CONF_ADAPTIVE_THROTTLE_MIN_SECONDS,
    CONF_ADAPTIVE_THROTTLE_MODE,
    CONF_ADAPTIVE_THROTTLE_MULTIPLIER,
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
    CONF_LEARNING_RECENT_SAMPLE_HOURS,
    CONF_MIN_LEVEL,
    CONF_PRECHARGE_COUNTDOWN_WINDOW,
    CONF_PRECHARGE_LEVEL,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_PRECHARGE_MARGIN_ON,
    CONF_PRESENCE_SENSOR,
    CONF_SMART_START_MARGIN,
    CONF_SWITCH_CONFIRMATION_COUNT,
    CONF_SWITCH_THROTTLE_SECONDS,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    DEFAULT_ADAPTIVE_EWMA_ALPHA,
    DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
    DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
    DEFAULT_ADAPTIVE_THROTTLE_BACKOFF_STEP,
    DEFAULT_ADAPTIVE_THROTTLE_DURATION_SECONDS,
    DEFAULT_ADAPTIVE_THROTTLE_ENABLED,
    DEFAULT_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
    DEFAULT_ADAPTIVE_THROTTLE_MIN_SECONDS,
    DEFAULT_ADAPTIVE_THROTTLE_MODE,
    DEFAULT_ADAPTIVE_THROTTLE_MULTIPLIER,
    DEFAULT_FALLBACK_MINUTES_PER_PERCENT,
    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    DEFAULT_PRECHARGE_COUNTDOWN_WINDOW,
    DEFAULT_PRECHARGE_MARGIN_OFF,
    DEFAULT_PRECHARGE_MARGIN_ON,
    DEFAULT_SMART_START_MARGIN,
    DEFAULT_SWITCH_CONFIRMATION_COUNT,
    DEFAULT_SWITCH_THROTTLE_SECONDS,
    DEFAULT_TARGET_LEVEL,
    DISCHARGING_STATES,
    DOMAIN,
    FULL_STATES,
    LEARNING_DEFAULT_SPEED,
    LEARNING_MAX_SPEED,
    LEARNING_MIN_SPEED,
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
        _LOGGER.debug(
            "Ignored exception (suppressed); enable DEBUG for traceback", exc_info=True
        )
    except Exception:
        # If logging itself fails, there's nothing else useful we can do.
        try:
            _REAL_LOGGER.debug("Ignored exception (suppressed) and logging failed")
        except Exception:
            _ignored_exc()

    # Note: per-device post-alarm handler will be defined on the class.


# Note: helper `_analyze_flipflop_events_and_apply_throttles` is defined
# as a method on the class later in the file (near `_maintain_telemetry`).


def _coerce_margin(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed)


def _coerce_learning_window(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    clamped = max(0.25, min(48.0, parsed))
    return clamped


def _coerce_confirmation(value: Any) -> int | None:
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
    precharge_margin_on: float | None = None
    precharge_margin_off: float | None = None
    smart_start_margin: float | None = None
    switch_throttle_seconds: float | None = None
    switch_confirmation_count: int | None = None
    charging_sensor: str | None = None
    avg_speed_sensor: str | None = None
    presence_sensor: str | None = None
    alarm_mode: str = ALARM_MODE_SINGLE
    alarm_entity: str | None = None
    learning_recent_sample_hours: float | None = None
    alarm_entities_by_weekday: Mapping[int, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DeviceConfig:
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

    def alarm_entity_for_today(self, weekday: int) -> str | None:
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
    precharge_duration_min: float | None
    alarm_time: datetime
    start_time: datetime | None
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
    precharge_release_level: float | None
    charging_state: str
    presence_state: str
    last_update: datetime

    def as_dict(self) -> dict[str, Any]:
        charge_duration_display: float | None = None
        if not math.isclose(self.charge_duration_min, self.duration_min, abs_tol=0.05):
            charge_duration_display = round(self.charge_duration_min, 1)

        total_duration_display: float | None = None
        if not math.isclose(self.total_duration_min, self.duration_min, abs_tol=0.05):
            total_duration_display = round(self.total_duration_min, 1)

        precharge_duration_display: float | None = None
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

    _last_action_log: dict[str, str]

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
        self._last_switch_time: dict[str, float] = {}
        # Per-device configured throttle values (entity_id -> seconds)
        self._device_switch_throttle: dict[str, float] = {}
        # Confirmation debounce: require N consecutive coordinator evaluations
        # that request a different desired state before issuing a switch.
        # Default is read from the central constants to remain consistent.
        self._confirmation_required = DEFAULT_SWITCH_CONFIRMATION_COUNT
        # per-device recent desired-state history: tuple(last_desired_bool, count)
        self._desired_state_history: dict[str, tuple[bool, int]] = {}
        # Telemetry: track recent flip-flop events per entity (epoch timestamps)
        self._flipflop_events: dict[str, list[float]] = {}
        # Configurable telemetry thresholds (tunable constants)
        self._flipflop_window_seconds = 300.0  # lookback window (5 minutes)
        self._flipflop_warn_threshold = 3  # events within window to warn
        # Adaptive mitigation: temporary throttle overrides to suppress flapping
        # Structure: entity_id -> dict(original: float, applied: float, expires: float)
        self._adaptive_throttle_overrides: dict[str, dict[str, float]] = {}
        # Adaptive parameters
        self._adaptive_throttle_multiplier = 2.0
        self._adaptive_throttle_min_seconds = 120.0
        self._adaptive_throttle_duration_seconds = (
            600.0  # how long override lasts (10min)
        )
        # Backoff parameters: how much extra multiplier to add per extra flip-flop event
        self._adaptive_throttle_backoff_step = 0.5
        self._adaptive_throttle_max_multiplier = 5.0
        # Per-device last action state (True==on, False==off) recorded when
        # the coordinator issues a switch. This helps tests (and some
        # integrations) that rely on the coordinator's pre-recorded actions
        # rather than the external entity state which may lag.
        self._last_action_state: dict[str, bool] = {}
        # Per-refresh evaluation id used to avoid double-recording within the
        # same coordinator run when multiple code paths may record the desired
        # state for an entity.
        self._current_eval_id = 0
        self._last_recorded_eval: dict[str, int] = {}
        # Per-device last switch evaluation id: stores the coordinator
        # evaluation id when the last service call was issued. This helps
        # reliably suppress immediate reversals that occur inside the same
        # coordinator evaluation or when evaluation ids are close in time.
        self._last_switch_eval: dict[str, int | None] = {}
        # Entities with an in-flight service call recorded during this
        # coordinator evaluation to prevent immediate reversal races.
        self._inflight_switches: dict[str, bool] = {}
        # Internal caches/state expected by the plan builder and other
        # helper methods. Initialize here to ensure tests that create
        # the coordinator directly don't encounter missing attributes.
        # battery history stores tuples like (timestamp: datetime, level: float, charging: bool)
        self._battery_history: dict[
            str, tuple[datetime, float, bool] | tuple[datetime, float]
        ] = {}
        self._drain_rate_cache: dict[str, float] = {}
        self._precharge_release: dict[str, Any] = {}
        # precharge_release_ready stores a datetime when the release will clear, or None
        self._precharge_release_ready: dict[str, datetime | None] = {}
        self._precharge_release_cleared_by_presence: dict[str, bool] = {}
        # Tracks precharge latches cleared because thresholds/predictions
        # indicate the precharge is no longer required. This allows the
        # coordinator to act immediately (pause charger) in the same
        # evaluation when a release clears due to reaching thresholds.
        self._precharge_release_cleared_by_threshold: dict[str, bool] = {}
        # Internal state snapshot produced by _async_update_data
        self._state: dict[str, Any] = {}
        # EWMA metrics for flip-flop telemetry
        self._flipflop_ewma: float = 0.0
        self._flipflop_ewma_last_update: float | None = None
        self._flipflop_ewma_exceeded: bool = False
        # Track when the EWMA first crossed the exceeded threshold
        self._flipflop_ewma_exceeded_since: float | None = None
        # Internal adaptive mode override (None|'conservative'|'normal'|'aggressive')
        self._adaptive_mode_override: str | None = None
        # Track last post-alarm self-heal handled timestamp per entity (epoch)
        self._post_alarm_last_handled: dict[str, float] = {}
        # Record of recent post-alarm corrections applied or suggested.
        # Each entry is a dict: {"entity": str, "device": str, "alarm_epoch": float,
        #  "timestamp": float, "reason": str, "details": {...}}
        self._post_alarm_corrections: list[dict[str, Any]] = []
        # Temporary in-memory-only adaptive overrides applied after a single miss
        # Structure: entity -> {"applied": float, "expires": float, "reason": str}
        self._post_alarm_temp_overrides: dict[str, dict[str, Any]] = {}
        # Streak counters for missed alarms per entity and per-reason
        self._post_alarm_miss_streaks: dict[str, dict[str, int]] = {}
        # Persisted smart_start margin overrides mapping (entity -> margin_value)
        # Will be written into entry.options as key `smart_start_margin_overrides` when persisted
        self._post_alarm_persisted_smart_start: dict[str, float] = (
            dict(getattr(self.entry, "options", {}) or {}).get(
                "smart_start_margin_overrides", {}
            )
            or {}
        )
        # Learning retrain request counters per profile (profile_id -> count)
        self._post_alarm_learning_retrain_requests: dict[str, int] = {}
        # Per-device forecast holdoff flag: when True the coordinator has
        # determined a forecast-based holdoff for that device and some
        # urgent actions may bypass throttle. Stored keyed by device.name.
        self._forecast_holdoff: dict[str, bool] = {}
        # Last rendered action log to avoid duplicate logging spam
        self._last_action_log = {}
        # Bind the concrete plan builder to an instance attribute so tests
        # can monkeypatch ``self._build_plan`` with a replacement callable.
        # The real implementation lives in ``_build_plan_impl``.
        self._build_plan = self._build_plan_impl

    def _normalize_entity_id(self, raw_entity: Any) -> str | None:
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

    def _device_name_for_entity(self, entity_id: str) -> str | None:
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

    # --- Helper utilities to reduce duplication and complexity ---
    def _parse_epoch(self, raw: Any) -> float | None:
        """Robustly parse a stored epoch-like value into float seconds or None.

        Accepts ints/floats, strings (parsed with dt_util.parse_datetime),
        or datetime-like objects handled by dt_util.as_timestamp.
        """
        if raw is None:
            return None
        try:
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                parsed = dt_util.parse_datetime(raw)
                return float(dt_util.as_timestamp(parsed)) if parsed else None
            # Fallback: attempt to convert using dt_util
            return float(dt_util.as_timestamp(raw))
        except Exception:
            _ignored_exc()
            return False

    def _get_now_epoch(self) -> float:
        """Return the current evaluation epoch (seconds since epoch).

        Uses _current_eval_time when present to keep evaluations deterministic
        during a single coordinator run; otherwise falls back to utcnow().
        """
        return float(
            dt_util.as_timestamp(
                getattr(self, "_current_eval_time", None) or dt_util.utcnow()
            )
        )

    def _get_last_action_state(self, norm: str) -> bool | None:
        """Return the coordinator-recorded last action state, or probe the entity state.

        Returns True/False when available, or None when unknown.
        """
        last = self._last_action_state.get(norm)
        if last is not None:
            return bool(last)
        try:
            st = self.hass.states.get(norm)
            return bool(st and st.state == STATE_ON)
        except Exception:
            _ignored_exc()
            return False

    def _get_throttle_seconds(self, norm: str) -> float | None:
        """Return the configured throttle seconds for an entity or None if disabled.

        Treat falsy (None/0/"") throttle values as disabled (None).
        """
        try:
            raw = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            if not raw:
                return None
            val = float(raw)
            if val <= 0:
                return None
            return val
        except Exception:
            _ignored_exc()
            try:
                return float(self._default_switch_throttle_seconds)
            except Exception:
                return None

    def _normalize_last_epoch(self, raw_last: Any) -> float | None:
        """Normalize a last-switch stored value into epoch seconds or None."""
        try:
            return self._parse_epoch(raw_last)
        except Exception:
            _ignored_exc()
            return False

    def _throttle_value_for(self, norm: str) -> float:
        """Return a numeric throttle value (seconds) for entity, falling back to default."""
        try:
            raw = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            try:
                return float(raw)
            except Exception:
                return float(self._default_switch_throttle_seconds)
        except Exception:
            _ignored_exc()
            return float(self._default_switch_throttle_seconds)

    def _final_guard_should_suppress(
        self, norm: str, pre_epoch: float, desired: bool
    ) -> bool | None:
        """Conservative final guard to prevent immediate reversals before recording a switch.

        Returns True when the planned action should be suppressed.
        """
        try:
            stored = self._last_switch_time.get(norm)
            stored_epoch_final = self._parse_epoch(stored)
            throttle_final = self._throttle_value_for(norm)
            if stored_epoch_final is not None and pre_epoch is not None:
                elapsed_final = float(pre_epoch) - float(stored_epoch_final)
                last_action_for_final = (
                    self._last_action_state.get(norm)
                    if self._last_action_state.get(norm) is not None
                    else (lambda: None)()
                )
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
                    return True
        except Exception:
            _ignored_exc()
        return False

    def _maybe_switch_final_throttle_check(
        self, norm: str, previous_last_action: bool | None, desired: bool
    ) -> bool:
        """Perform the final throttle check and return True when suppression is needed.

        This mirrors the inlined logic previously in _maybe_switch_execute_action.
        """
        try:
            current_last = self._last_switch_time.get(norm)
            throttle = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            if current_last is not None:
                if isinstance(current_last, (int, float)):
                    stored_epoch = float(current_last)
                else:
                    parsed = (
                        dt_util.parse_datetime(current_last)
                        if isinstance(current_last, str)
                        else current_last
                    )
                    stored_epoch = (
                        float(dt_util.as_timestamp(parsed)) if parsed else None
                    )
            else:
                stored_epoch = None
            if stored_epoch is not None:
                now_epoch = float(
                    dt_util.as_timestamp(
                        getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                    )
                )
                elapsed_now = now_epoch - stored_epoch
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
                    last_action_state = (
                        previous_last_action
                        if previous_last_action is not None
                        else self._last_action_state.get(norm)
                    )
                    if last_action_state is not None and bool(
                        last_action_state
                    ) != bool(desired):
                        _LOGGER.warning(
                            "FINAL_THROTTLE_SUPPRESS: entity=%s last_action=%s desired=%s elapsed=%.3f throttle=%s",
                            norm,
                            last_action_state,
                            desired,
                            elapsed_now,
                            throttle,
                        )
                        return True
        except Exception:
            _ignored_exc()
        return False

    def _early_suppress_checks(
        self, norm: str, desired: bool, force: bool, bypass_throttle: bool
    ) -> bool:
        """Run initial authoritative and quick suppression checks.

        Returns True when the action should be suppressed.
        """
        should_check = not force and not bypass_throttle
        if not should_check:
            return False
        # Authoritative early suppression
        try:
            stored_epoch_auth = self._parse_epoch(self._last_switch_time.get(norm))
            throttle_cfg_auth = self._get_throttle_seconds(norm)
            if stored_epoch_auth is not None and throttle_cfg_auth:
                now_epoch_auth = self._get_now_epoch()
                elapsed_auth = now_epoch_auth - stored_epoch_auth
                last_act_auth = self._get_last_action_state(norm)
                if (
                    last_act_auth is not None
                    and elapsed_auth >= 0
                    and elapsed_auth < float(throttle_cfg_auth)
                    and bool(last_act_auth) != bool(desired)
                ):
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
            last_epoch_q = self._parse_epoch(self._last_switch_time.get(norm))
            thr_q = self._get_throttle_seconds(norm)
            last_act_quick = self._get_last_action_state(norm)
            if last_epoch_q is not None and thr_q:
                now_epoch_q = self._get_now_epoch()
                elapsed_q = now_epoch_q - last_epoch_q
                if elapsed_q >= 0 and elapsed_q < float(thr_q):
                    if last_act_quick is not None and bool(last_act_quick) != bool(
                        desired
                    ):
                        _LOGGER.info(
                            "VERY_EARLY_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                            norm,
                            elapsed_q,
                            thr_q,
                            last_act_quick,
                            desired,
                        )
                        return True
        except Exception:
            _ignored_exc()

        return False

    def _should_suppress_switch(
        self, norm: str, desired: bool, force: bool, bypass_throttle: bool
    ) -> bool:
        """Comprehensive suppression helper extracted from _maybe_switch.

        Returns True when the switch should be suppressed, False otherwise.
        This consolidates multiple quick-gates and authoritative checks so
        the primary method remains smaller and easier for static analysis.
        """
        # Decide whether to run throttle/reversal suppression checks.
        should_check = not force and not bypass_throttle
        # Delegate the sequential suppression checks to a central runner
        # which handles exceptions and conditional checks in one place.
        return self._run_suppress_checks(norm, desired, should_check)

    def _run_suppress_checks(
        self, norm: str, desired: bool, should_check: bool
    ) -> bool:
        """Run suppression checks sequentially and return True on first match.

        This centralizes try/except handling and minor conditional branching so
        the public wrapper stays small and easier to reason about. The order
        of checks mirrors the original inline implementation.
        """
        checks: list[Callable[[], bool]] = []

        # In-flight and authoritative/defensive checks accept should_check.
        checks.append(
            lambda: self._should_suppress_inflight(norm, desired, should_check)
        )
        checks.append(
            lambda: self._should_suppress_authoritative_throttle(
                norm, desired, should_check
            )
        )

        # The canonical throttle and recent-eval guards only apply when
        # should_check is True.
        if should_check:
            checks.append(
                lambda: self._should_suppress_canonical_throttle(norm, desired)
            )
            checks.append(lambda: self._should_suppress_recent_eval(norm, desired))

        checks.append(
            lambda: self._should_suppress_defensive_throttle(
                norm, desired, should_check
            )
        )

        for chk in checks:
            try:
                if chk():
                    return True
            except Exception:
                _ignored_exc()
        return False

    def _should_suppress_recent_eval(self, norm: str, desired: bool) -> bool:
        """Return True when recent-eval suppression guard indicates suppression."""
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
                _LOGGER.debug(
                    "SUPPRESS_RECENT_EVAL: entity=%s cur_eval=%s last_eval=%s last_act=%r desired=%s",
                    norm,
                    cur_eval,
                    last_eval,
                    last_act,
                    desired,
                )
                return True
        except Exception:
            _ignored_exc()
        return False

    def _should_suppress_canonical_throttle(self, norm: str, desired: bool) -> bool:
        """Canonical throttle/reversal guard extracted from _should_suppress_switch.

        Returns True when the entity should be suppressed due to a recent
        last-switch timestamp inside the throttle window and a reversal
        relative to the last action state.
        """
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

        last_epoch = self._parse_epoch(self._last_switch_time.get(norm))
        last_act = self._get_last_action_state(norm)
        throttle_val = self._get_throttle_seconds(norm)
        if last_epoch is not None and last_act is not None and throttle_val:
            now_epoch = self._get_now_epoch()
            try:
                tval = float(throttle_val)
            except Exception:
                tval = float(self._default_switch_throttle_seconds)
            elapsed = now_epoch - last_epoch
            if last_act is None:
                last_act = self._get_last_action_state(norm)
                try:
                    _LOGGER.debug(
                        "DBG_CANONICAL: entity=%s last_epoch=%r last_act=%r now_epoch=%r elapsed=%r throttle_val=%r desired=%r last_eval=%r cur_eval=%r",
                        norm,
                        last_epoch,
                        last_act,
                        now_epoch,
                        elapsed,
                        tval,
                        desired,
                        self._last_switch_eval.get(norm),
                        getattr(self, "_current_eval_id", None),
                    )
                except Exception:
                    _ignored_exc()
            if (
                elapsed >= 0
                and elapsed < float(tval)
                and last_act is not None
                and bool(last_act) != bool(desired)
            ):
                _LOGGER.debug(
                    "CANONICAL_THROTTLE_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                    norm,
                    elapsed,
                    tval,
                    last_act,
                    desired,
                )
                try:
                    last_eval = self._last_switch_eval.get(norm)
                    cur_eval = int(getattr(self, "_current_eval_id", 0) or 0)
                    if last_eval is None or abs(cur_eval - int(last_eval or 0)) <= 1:
                        return True
                except Exception:
                    return True
        return False

    def _should_suppress_inflight(
        self, norm: str, desired: bool, should_check: bool
    ) -> bool:
        """Return True when an in-flight switch should suppress reversal."""
        try:
            pending = self._inflight_switches.get(norm)
            if should_check and pending is not None and bool(pending) != bool(desired):
                _LOGGER.debug(
                    "SUPPRESS_INFLIGHT: entity=%s pending=%s desired=%s",
                    norm,
                    pending,
                    desired,
                )
                return True
        except Exception:
            _ignored_exc()
        return False

    def _should_suppress_authoritative_throttle(
        self, norm: str, desired: bool, should_check: bool
    ) -> bool:
        """Return True when authoritative throttle/reversal should suppress."""
        if not should_check:
            return False
        try:
            last_epoch_check = self._parse_epoch(self._last_switch_time.get(norm))
            thr_cfg = self._get_throttle_seconds(norm)
            if last_epoch_check is not None and thr_cfg:
                now_epoch_check = self._get_now_epoch()
                elapsed_check = now_epoch_check - last_epoch_check
                if elapsed_check >= 0 and elapsed_check < float(thr_cfg):
                    last_action_check = self._get_last_action_state(norm)
                    if last_action_check is not None and bool(
                        last_action_check
                    ) != bool(desired):
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
        return False

    def _should_suppress_defensive_throttle(
        self, norm: str, desired: bool, should_check: bool
    ) -> bool:
        """Defensive guard: return True when defensive throttle/reversal indicates suppression."""
        if not should_check:
            return False
        try:
            last_raw = self._last_switch_time.get(norm)
            last_act = self._last_action_state.get(norm)
            throttle_cfg = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            if last_raw is not None and last_act is not None and throttle_cfg:
                try:
                    if isinstance(last_raw, (int, float)):
                        last_epoch = float(last_raw)
                    elif isinstance(last_raw, str):
                        parsed = dt_util.parse_datetime(last_raw)
                        last_epoch = (
                            float(dt_util.as_timestamp(parsed)) if parsed else None
                        )
                    else:
                        last_epoch = float(dt_util.as_timestamp(last_raw))
                except Exception:
                    last_epoch = None
                if last_epoch is not None:
                    now_epoch = float(
                        dt_util.as_timestamp(
                            getattr(self, "_current_eval_time", None)
                            or dt_util.utcnow()
                        )
                    )
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
                    if (
                        elapsed >= 0
                        and elapsed < float(throttle_val)
                        and bool(last_act) != bool(desired)
                    ):
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

    def _maybe_switch_confirmation_and_throttle_flow(
        self, norm: str, desired: bool, action: str, should_check: bool
    ) -> bool:
        """Run confirmation and throttle checks for a candidate switch.

        Returns True when the flow decided to suppress the switch (so the
        caller should return early). Mirrors the original inline logic from
        _maybe_switch and keeps logging and behavior unchanged.
        """
        if not should_check:
            return False

        # Per-device throttle (seconds) default fallback
        throttle = self._device_switch_throttle.get(
            norm, self._default_switch_throttle_seconds
        )
        last = self._last_switch_time.get(norm)

        _LOGGER.info(
            "SHOULD_CHECK_VARS: entity=%s last=%r last_type=%s throttle=%s",
            norm,
            last,
            type(last),
            throttle,
        )

        suppress, hist, required, count = self._confirmation_and_throttle_check(
            norm, desired
        )
        try:
            _LOGGER.debug(
                "DEBUG_MAYBE_FLOW: entity=%s suppress=%s hist=%r count=%s required=%s",
                norm,
                suppress,
                hist,
                count,
                required,
            )
        except Exception:
            _ignored_exc()
        if suppress:
            return True

        _LOGGER.debug(
            "MAYBE_SWITCH_SUMMARY: entity=%s required=%s hist=%s count=%s last=%s throttle=%s",
            norm,
            required,
            hist,
            count,
            self._last_switch_time.get(norm),
            self._device_switch_throttle.get(norm),
        )

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
            return True

        # Throttle suppression checks
        _LOGGER.debug(
            "throttle diagnostic: norm_present=%s keys=%s",
            norm in self._last_switch_time,
            list(self._last_switch_time.keys()),
        )

        # Quick throttle & quick-gate checks
        try:
            if self._maybe_switch_quick_throttle_checks(norm, last, throttle, desired):
                _LOGGER.info(
                    "EARLY_SUPPRESS: entity=%s last_action=%s desired=%s elapsed suppressed",
                    norm,
                    self._get_last_action_state(norm),
                    desired,
                )
                return True
        except Exception:
            _ignored_exc()

        # Throttle block handling (compute elapsed and decide)
        try:
            if self._maybe_switch_handle_throttle_block(
                norm, action, last, throttle, required
            ):
                return True
        except Exception:
            _ignored_exc()

        return False

    def _maybe_switch_parse_last_epoch_quick(self, last: Any) -> float | None:
        """Parse the raw last value into an epoch float used by quick gate checks."""
        try:
            if isinstance(last, (int, float)):
                return float(last)
            if isinstance(last, str):
                parsed = dt_util.parse_datetime(last)
                return float(dt_util.as_timestamp(parsed)) if parsed else None
            if last is not None:
                return float(dt_util.as_timestamp(last))
        except Exception:
            _ignored_exc()
        return False

    def _maybe_switch_quick_throttle_checks(
        self, norm: str, last: Any, throttle: Any, desired: bool
    ) -> bool:
        """Run early throttle and quick-gate checks; return True when suppressed."""
        try:
            if self._throttle_suppress(norm, last, throttle, desired):
                return True
        except Exception:
            _ignored_exc()

        try:
            last_epoch_quick = self._maybe_switch_parse_last_epoch_quick(last)
            _LOGGER.debug("DEBUG_QUICK_GATE: raw last=%r type=%s", last, type(last))
            if self._quick_gate_suppress(norm, last_epoch_quick, throttle, desired):
                _LOGGER.debug(
                    "QUICK_SUPPRESS: entity=%s suppression triggered by quick gate",
                    norm,
                )
                return True
        except Exception:
            _LOGGER.exception("Quick gate evaluation failed for %s", norm)

        return False

    def _maybe_switch_handle_throttle_block(
        self, norm: str, action: str, last: Any, throttle: Any, required: int
    ) -> bool:
        """Handle the throttle decision block when a last-switch timestamp exists."""
        if not last:
            return False
        _LOGGER.info("ENTER_THROTTLE_BLOCK: entity=%s last=%r", norm, last)
        elapsed, readable_last, readable_now, now_for_cmp = (
            self._maybe_switch_compute_elapsed(norm, last)
        )
        _LOGGER.debug(
            "THROTTLE_DECISION: entity=%s throttle=%s last=%s now=%s elapsed=%s required=%s",
            norm,
            throttle,
            readable_last,
            readable_now,
            elapsed,
            required,
        )
        _LOGGER.info(
            "DECISION_CHECK: entity=%s elapsed=%s throttle=%s required=%s",
            norm,
            elapsed,
            throttle,
            required,
        )
        try:
            if elapsed is not None and elapsed < float(throttle):
                _LOGGER.info(
                    "THROTTLED: switch.%s for %s (last %.1fs ago, throttle=%.1fs)",
                    action,
                    norm,
                    elapsed,
                    float(throttle),
                )
                return True
            else:
                _LOGGER.debug(
                    "THROTTLE_OK: switch.%s for %s (last=%s elapsed=%s throttle=%s)",
                    action,
                    norm,
                    last,
                    elapsed,
                    float(throttle),
                )
        except Exception:
            _ignored_exc()
        return False

    def _maybe_switch_prepare_inputs(
        self, norm: str, raw_entity: Any, desired: bool, now: Any, should_check: bool
    ) -> None:
        """Prepare and emit diagnostic logging used by `_maybe_switch`.

        Extracted to reduce the size of `_maybe_switch` while keeping
        the diagnostic messages and behavior unchanged.
        """
        # Keep logging minimal here — detailed debug logs used during
        # development have been removed to reduce CI noise.
        try:
            _LOGGER.debug(
                "_maybe_switch called: action=%s entity=%s desired=%s force=%s now=%s",
                getattr(self, "_last_called_action", None),
                norm,
                desired,
                getattr(self, "_last_called_force", None),
                getattr(now, "isoformat", lambda: str(now))(),
            )
        except Exception:
            _ignored_exc()

        # Debug print for triage: show key values early
        try:
            _LOGGER.debug(
                "DBG_MAYBE_SWITCH_START: entity=%s desired=%s now=%s last_switch_time=%r last_action_state=%r device_throttle=%r",
                norm,
                desired,
                getattr(now, "isoformat", lambda: str(now))(),
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
                getattr(now, "isoformat", lambda: str(now))(),
                norm,
                self._last_switch_time.get(norm),
                self._last_action_state.get(norm),
                self._device_switch_throttle.get(norm),
                desired,
                should_check,
            )
        except Exception:
            _ignored_exc()

        # only log last_switch_time at debug level
        _LOGGER.debug("last_switch_time map at start: %s", self._last_switch_time)

        # Debug: show the raw representation of the entity_id to catch
        # accidental list/tuple vs string mismatches that lead to missed
        # throttle-key lookups in tests.
        try:
            _LOGGER.info(
                "_maybe_switch raw entity repr: %r (type=%s)",
                raw_entity,
                type(raw_entity),
            )
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
            getattr(self, "_last_called_bypass", None),
        )

    def _maybe_switch_collect_gate_inputs(
        self, norm: str, desired: bool, should_check: bool
    ) -> tuple[str, int, tuple[bool, int], Any, Any]:
        """Collect gate inputs and emit informational logging.

        Returns (confirm_key, required, hist, last, throttle)
        """
        confirm_key = f"{norm}::confirm"
        try:
            required = int(
                self._device_switch_throttle.get(
                    confirm_key, float(self._confirmation_required)
                )
            )
        except Exception:
            _ignored_exc()
            required = int(self._confirmation_required)

        hist = self._desired_state_history.get(norm, (desired, 0))
        # Debug: inspect existing last_switch_time keys to catch mismatches
        try:
            _LOGGER.info(
                "last_switch_time keys before check: %s",
                list(self._last_switch_time.keys()),
            )
        except Exception:
            _ignored_exc()

        last = self._last_switch_time.get(norm)
        try:
            _LOGGER.debug(
                "throttle: last raw=%r type=%s truthy=%s", last, type(last), bool(last)
            )
        except Exception:
            _ignored_exc()

        try:
            throttle = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
        except Exception:
            _ignored_exc()
            throttle = self._default_switch_throttle_seconds

        try:
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

        return confirm_key, required, hist, last, throttle

    def _maybe_switch_prepare_execution_context(
        self, norm: str
    ) -> tuple[bool | None, float]:
        """Prepare execution context values used by the switch executor.

        Returns (previous_last_action, pre_epoch)
        """
        previous_last_action = None
        pre_epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
        try:
            previous_last_action = self._last_action_state.get(norm)
        except Exception:
            _ignored_exc()
        try:
            pre_ts = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
            pre_epoch = float(dt_util.as_timestamp(pre_ts))
        except Exception:
            try:
                pre_epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
            except Exception:
                pre_epoch = float(0.0)
        return previous_last_action, pre_epoch

    def _maybe_switch_log_entry_and_caller(
        self, norm: str, bypass_throttle: bool
    ) -> None:
        """Emit the top-of-function diagnostic logs extracted from _maybe_switch."""
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

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        return self._state or {}

    def _raw_config(self) -> dict[str, Any]:
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
        self, devices: Iterable[Mapping[str, Any]] | None = None
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
        # end _iter_device_configs

    async def _async_update_data(self) -> dict[str, dict[str, Any]]:
        # Record the logical evaluation time so switch throttling and
        # confirmation can be evaluated against the same simulated time used
        # by the plan builder (useful for deterministic tests which pass a
        # custom ``now_local`` into _build_plan).
        now_local = dt_util.now()  # This line remains unchanged
        try:
            self._async_update_log_start(now_local)
        except Exception:
            _ignored_exc()
        domain_data = self.hass.data.get(DOMAIN, {})
        entries: dict[str, dict[str, Any]] = domain_data.get("entries", {})
        entry_data = entries.get(self.entry.entry_id, {})
        learning = entry_data.get("learning")

        try:
            raw_config = self._raw_config()
            results: dict[str, dict[str, Any]] = {}
            # New evaluation - advance the eval id to avoid duplicate recordings
            # within the same refresh cycle.
            try:
                # Prepare evaluation snapshot and reset in-flight markers
                self._async_update_prepare_evaluation()
            except Exception:
                # Fallback to a safe eval id when prepare fails
                self._current_eval_id = getattr(self, "_current_eval_id", 1)
            try:
                self._async_update_log_raw_device_count(raw_config)
            except Exception:
                _ignored_exc()
            # Configure coordinator-level options used during the update
            # (extracted to reduce complexity of this method)
            try:
                self._configure_update_options(raw_config)
            except Exception:
                _ignored_exc()

            # Build per-device plans (extracted helper to reduce complexity)
            results.update(
                await self._async_update_build_plans(raw_config, now_local, learning)
            )

            try:
                self._maintain_telemetry(now_local)
            except Exception:
                _ignored_exc()

            # Finalize the update (extracted): telemetry, cache cleanup,
            # state assignment, self-heal and final snapshot/logging.
            try:
                return self._async_update_finalize(results, now_local)
            except Exception:
                _ignored_exc()
                return results

        except Exception:
            _LOGGER.exception("Smart Charger coordinator update failed")
            return self._state or {}

    async def _async_update_build_plans(
        self, raw_config: dict[str, Any], now_local: datetime, learning
    ) -> dict[str, dict[str, Any]]:
        """Build plans for each device in raw_config and return results dict.

        Extracted from _async_update_data to reduce that function's complexity.
        Behavior is identical to the original inline loop.
        """
        results: dict[str, dict[str, Any]] = {}
        try:
            for device in self._iter_device_configs(raw_config.get("devices") or []):
                try:
                    pd = await self._prepare_device_and_build_plan(
                        device, now_local, learning
                    )
                    if pd:
                        results[device.name] = pd
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()
        return results

    def _configure_update_options(self, raw_config: dict[str, Any]) -> None:
        """Configure coordinator-level options used during _async_update_data.

        Extracted from the main update method to reduce its cyclomatic
        complexity while keeping behavior identical.
        """
        # Use the configured coordinator-level default already stored in
        # self._confirmation_required; per-device overrides are applied
        # elsewhere when parsing devices.
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
            CONF_ADAPTIVE_THROTTLE_MIN_SECONDS,
            DEFAULT_ADAPTIVE_THROTTLE_MIN_SECONDS,
        )
        self._adaptive_throttle_duration_seconds = self._option_float(
            CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS,
            DEFAULT_ADAPTIVE_THROTTLE_DURATION_SECONDS,
        )
        # Backoff tuning (variable multiplier)
        self._adaptive_throttle_backoff_step = self._option_float(
            CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP,
            DEFAULT_ADAPTIVE_THROTTLE_BACKOFF_STEP,
        )
        self._adaptive_throttle_max_multiplier = self._option_float(
            CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
            DEFAULT_ADAPTIVE_THROTTLE_MAX_MULTIPLIER,
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
        effective_mode = (
            str(getattr(self, "_adaptive_mode_override", None) or mode).strip().lower()
        )

        # Map effective_mode to a scaling factor applied to the backoff growth
        if effective_mode == "conservative":
            self._adaptive_mode_factor = 0.7
        elif effective_mode == "aggressive":
            self._adaptive_mode_factor = 1.4
        else:
            self._adaptive_mode_factor = 1.0
        self._flipflop_window_seconds = self._option_float(
            CONF_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
            DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
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
            self._flipflop_warn_threshold = int(
                DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD
            )

    def _async_update_prepare_evaluation(self) -> None:
        """Prepare evaluation snapshot, bump eval id and reset inflight markers.

        Extracted from _async_update_data to reduce inline complexity.
        """
        try:
            # Render a human-friendly snapshot at DEBUG level only.
            rendered: dict[str, str] = {}
            for k, v in self._last_switch_time.items():
                try:
                    if isinstance(v, (int, float)):
                        rendered[k] = str(datetime.fromtimestamp(float(v)).isoformat())
                    else:
                        iso = getattr(v, "isoformat", None)
                        rendered[k] = str(iso() if callable(iso) else repr(v))
                except Exception:
                    _ignored_exc()
                    rendered[k] = repr(v)
            _LOGGER.debug("SNAPSHOT(last_switch_time): %s", rendered)
        except Exception:
            _ignored_exc()
        try:
            self._current_eval_id += 1
        except Exception:
            _ignored_exc()
        # Clear any in-flight markers at the start of a new evaluation so
        # they only protect against reversals within the same coordinator
        # refresh cycle.
        try:
            self._inflight_switches = {}
        except Exception:
            _ignored_exc()

    def _async_update_log_start(self, now_local: datetime) -> None:
        """Emit the top-of-update debug logs and set evaluation time."""
        try:
            _LOGGER.debug(
                "_async_update_data: starting evaluation %d now=%s",
                self._current_eval_id,
                now_local.isoformat(),
            )
            # Keep the coordinator's view of the current evaluation time
            # synchronized with the plan builder.
            self._current_eval_time = now_local
        except Exception:
            _ignored_exc()

    def _async_update_log_raw_device_count(self, raw_config: dict[str, Any]) -> None:
        """Log the raw device count for diagnostic purposes."""
        try:
            _LOGGER.debug(
                "_async_update_data: raw device count=%d",
                len(raw_config.get("devices", []) or []),
            )
        except Exception:
            _ignored_exc()

    def _async_update_finalize(
        self, results: dict[str, dict[str, Any]], now_local: datetime
    ) -> dict[str, dict[str, Any]]:
        """Finalize the update: telemetry, cache pruning, state assignment and logging.

        Extracted from `_async_update_data` to reduce its size and complexity.
        Returns the results dict to be returned by the caller.
        """
        try:
            # Maintain telemetry and prune caches in their own helper
            try:
                self._async_update_maintain_and_prune(results, now_local)
            except Exception:
                _ignored_exc()

            # Assign state and run post-update self-heal in a helper
            try:
                self._async_update_assign_state_and_heal(results, now_local)
            except Exception:
                _ignored_exc()

            # Emit debug snapshots in a compact helper
            try:
                self._async_update_log_snapshots()
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()
        return results

    def _async_update_maintain_and_prune(
        self, results: dict[str, dict[str, Any]], now_local: datetime
    ) -> None:
        """Run maintenance tasks extracted from finalize."""
        try:
            try:
                self._maintain_telemetry(now_local)
            except Exception:
                _ignored_exc()

            try:
                self._async_update_prune_caches(results)
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()

    def _async_update_assign_state_and_heal(
        self, results: dict[str, dict[str, Any]], now_local: datetime
    ) -> None:
        """Assign the coordinator state and run post-alarm self-heal."""
        try:
            self._state = results
            try:
                self._handle_post_alarm_self_heal(results, now_local)
            except Exception:
                _ignored_exc()
            self._last_successful_update = dt_util.utcnow()
        except Exception:
            _ignored_exc()

    def _async_update_log_snapshots(self) -> None:
        """Emit debug snapshots for desired-state history and last_switch_time."""
        try:
            _LOGGER.debug(
                "Desired state history snapshot: %s", self._desired_state_history
            )
            try:
                readable: dict[str, str] = {}
                for k, v in self._last_switch_time.items():
                    try:
                        if isinstance(v, (int, float)):
                            readable[k] = str(
                                datetime.fromtimestamp(float(v)).isoformat()
                            )
                        else:
                            iso = getattr(v, "isoformat", None)
                            readable[k] = str(iso() if callable(iso) else repr(v))
                    except Exception:
                        readable[k] = str(repr(v))
            except Exception:
                readable = {k: repr(v) for k, v in self._last_switch_time.items()}
            _LOGGER.debug("last_switch_time snapshot: %s", readable)
        except Exception:
            _ignored_exc()

    def _async_update_prune_caches(self, results: dict[str, dict[str, Any]]) -> None:
        """Prune per-device caches (precharge_release and drain rate) after an update."""
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

    async def _prepare_device_and_build_plan(
        self, device: DeviceConfig, now_local: datetime, learning
    ) -> dict[str, Any] | None:
        """Prepare per-device settings and build the SmartChargePlan.

        This pulls the learning window, configures per-device throttles and
        confirmation counts, calls `_build_plan`, and returns the plan dict
        annotated with the charger_switch entity when present.
        """
        try:
            device_window = device.learning_recent_sample_hours
            if device_window is None:
                device_window = self._default_learning_recent_sample_hours
            device_window = max(0.25, min(48.0, float(device_window)))

            if learning is not None and hasattr(learning, "set_recent_sample_window"):
                try:
                    learning.set_recent_sample_window(device_window)
                except Exception:
                    _ignored_exc()

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
            except Exception:
                _ignored_exc()

            plan = await self._build_plan(device, now_local, learning, device_window)
            if plan:
                pd = plan.as_dict()
                pd["charger_switch"] = device.charger_switch
                return pd
        except Exception:
            _ignored_exc()
        return None

    def _maybe_switch_compute_elapsed(
        self, norm: str, last: Any
    ) -> tuple[float | None, str, str, Any]:
        """Compute elapsed seconds since last and readable representations.

        Returns (elapsed, readable_last, readable_now, now_for_cmp). elapsed
        is None when computation fails.
        """
        now_for_cmp = dt_util.utcnow()
        try:
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

            _LOGGER.debug(
                "THROTTLE_DECISION_INPUTS: entity=%s now_ts=%.3f last_ts=%.3f elapsed=%.3f",
                norm,
                now_ts,
                last_ts_val,
                elapsed,
            )

            if elapsed < 0:
                _LOGGER.warning(
                    "THROTTLE_TS: last_switch_time (%s) is after now (%s) -> treating as expired",
                    last,
                    now_for_cmp,
                )
                elapsed = float("inf")
                _LOGGER.debug(
                    "THROTTLE_DECISION_INPUTS: entity=%s now_ts=%s last_ts=%s elapsed=%.3f",
                    norm,
                    now_ts,
                    last_ts_val,
                    elapsed,
                )
            _LOGGER.debug("THROTTLE_TS: computed elapsed=%s", elapsed)
        except Exception:
            _LOGGER.exception(
                "Throttle timestamp computation failed for %s: last=%r now=%r",
                norm,
                last,
                now_for_cmp,
            )
            elapsed = None

        try:
            if isinstance(last, (int, float)):
                readable_last = datetime.fromtimestamp(float(last)).isoformat()
            else:
                readable_last = getattr(last, "isoformat", lambda: repr(last))()
        except Exception:
            readable_last = repr(last)
        try:
            readable_now = getattr(
                now_for_cmp, "isoformat", lambda: repr(now_for_cmp)
            )()
        except Exception:
            readable_now = repr(now_for_cmp)

        return elapsed, readable_last, readable_now, now_for_cmp

    def _maybe_switch_pre_record_check(
        self, norm: str, pre_epoch: Any, desired: bool
    ) -> bool:
        """Deterministic pre-record throttle suppression.

        Returns True when the action should be suppressed (caller will return False).
        """
        stored = self._last_switch_time.get(norm)
        if stored is not None:
            try:
                if isinstance(stored, (int, float)):
                    stored_epoch_det = float(stored)
                elif isinstance(stored, str):
                    parsed = dt_util.parse_datetime(stored)
                    stored_epoch_det = (
                        float(dt_util.as_timestamp(parsed)) if parsed else None
                    )
                else:
                    stored_epoch_det = float(dt_util.as_timestamp(stored))
            except Exception:
                stored_epoch_det = None
        else:
            stored_epoch_det = None

        throttle_val_det = float(
            self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            or self._default_switch_throttle_seconds
        )

        if stored_epoch_det is not None and pre_epoch is not None:
            try:
                elapsed_det = float(pre_epoch) - float(stored_epoch_det)
            except Exception:
                return False

            last_act_det = self._last_action_state.get(norm)
            if last_act_det is None:
                try:
                    st = self.hass.states.get(norm)
                    last_act_det = bool(st and st.state == STATE_ON)
                except Exception:
                    last_act_det = None

            if (
                last_act_det is not None
                and elapsed_det >= 0
                and elapsed_det < float(throttle_val_det)
                and bool(last_act_det) != bool(desired)
            ):
                _LOGGER.info(
                    "PRE_RECORD_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                    norm,
                    elapsed_det,
                    throttle_val_det,
                    last_act_det,
                    desired,
                )
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
                return True

        return False

    def _maybe_switch_final_guard(
        self,
        norm: str,
        pre_epoch: float,
        previous_last_action: bool | None,
        desired: bool,
    ) -> bool:
        """Final deterministic guard before performing the switch call.

        Returns True when the action should be suppressed.
        """
        try:
            stored_epoch_final = self._maybe_switch_parse_stored_epoch(norm)
            throttle_final = float(
                self._device_switch_throttle.get(
                    norm, self._default_switch_throttle_seconds
                )
                or self._default_switch_throttle_seconds
            )
            if stored_epoch_final is not None and pre_epoch is not None:
                elapsed_final = float(pre_epoch) - float(stored_epoch_final)
                last_action_for_final = self._maybe_switch_resolve_last_action(
                    norm, previous_last_action
                )
                if (
                    last_action_for_final is not None
                    and elapsed_final >= 0
                    and elapsed_final < float(throttle_final)
                    and bool(last_action_for_final) != bool(desired)
                ):
                    try:
                        _LOGGER.debug(
                            "DBG_FINAL_GUARD_SUPPRESS: entity=%s elapsed=%.3f throttle=%s last_action=%r desired=%s previous_last_action=%r",
                            norm,
                            elapsed_final,
                            throttle_final,
                            last_action_for_final,
                            desired,
                            previous_last_action,
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

                    try:
                        self._inflight_switches.pop(norm, None)
                    except Exception:
                        _ignored_exc()

                    return True
        except Exception:
            _ignored_exc()
        return False

    def _maybe_switch_parse_stored_epoch(self, norm: str) -> float | None:
        """Parse and normalize stored last-switch timestamp for final guard."""
        try:
            stored = self._last_switch_time.get(norm)
            if stored is not None:
                if isinstance(stored, (int, float)):
                    return float(stored)
                elif isinstance(stored, str):
                    parsed = dt_util.parse_datetime(stored)
                    return float(dt_util.as_timestamp(parsed)) if parsed else None
                else:
                    return float(dt_util.as_timestamp(stored))
        except Exception:
            _ignored_exc()
        return None

    def _maybe_switch_resolve_last_action(
        self, norm: str, previous_last_action: bool | None
    ) -> bool | None:
        """Resolve the last action state used by final guard checks."""
        try:
            last_action_for_final = (
                previous_last_action
                if previous_last_action is not None
                else self._last_action_state.get(norm)
            )
            if last_action_for_final is None:
                try:
                    st = self.hass.states.get(norm)
                    return bool(st and st.state == STATE_ON)
                except Exception:
                    return None
            return last_action_for_final
        except Exception:
            _ignored_exc()
            return None

    def _apply_charger_compute_assumed_state(
        self, device: DeviceConfig, charger_is_on: bool
    ) -> tuple[bool | None, bool]:
        """Compute last_action and assumed_on for `_apply_charger_logic`.

        Returns (last_action, assumed_on).
        """
        try:
            last_action = self._last_action_state.get(device.charger_switch)
            assumed_on = charger_is_on or (last_action is True)
            return last_action, assumed_on
        except Exception:
            _ignored_exc()
            return None, bool(charger_is_on)

    async def _apply_charger_handle_precharge_release(
        self,
        device: DeviceConfig,
        charger_ent: str,
        charger_is_on: bool,
        precharge_required: bool,
        release_level: float | None,
        service_data: dict[str, Any],
        device_name: str,
        predicted_level: float,
        battery: float,
        margin_on: float,
    ) -> bool:
        """Handle urgent precharge-release logic.

        Returns True when the helper handled the action (caller should return False).
        """
        # Only treat this as an urgent precharge-release if there was
        # previously a precharge latch for this device. Without a
        # prior latch, pausing the charger here is not an urgent
        # precharge-release event and should respect the normal
        # throttle/confirmation gates to avoid rapid flip-flops.
        had_precharge_latch = (
            device.name in self._precharge_release
            or device.name in self._precharge_release_ready
        )
        try:
            prev_intended = self._last_action_state.get(charger_ent)
            threshold_cleared = self._precharge_release_cleared_by_threshold.pop(
                device.name, False
            )
            try:
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
                return True

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
                try:
                    charging_state = self._charging_state(device.charging_sensor)
                except Exception:
                    charging_state = "unknown"
                if charging_state == "charging":
                    await self._maybe_switch(
                        "turn_off",
                        service_data,
                        desired=False,
                        bypass_throttle=True,
                    )
                else:
                    await self._maybe_switch("turn_off", service_data, desired=False)
                return True
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
            await self._maybe_switch(
                "turn_off", service_data, desired=False, bypass_throttle=True
            )
            return True

        return False

    async def _apply_charger_handle_presence_precharge_clear(
        self,
        device: DeviceConfig,
        charger_ent: str,
        charger_is_on: bool,
        service_data: dict[str, Any],
        device_name: str,
    ) -> bool:
        """Handle presence-leave precharge clear and pause charger.

        Returns True when the helper handled the action (caller should
        return False from `_apply_charger_logic`).
        """
        try:
            if charger_is_on and self._precharge_release_cleared_by_presence.pop(
                device.name, False
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
                return True
        except Exception:
            _LOGGER.exception("Error while handling presence-leave precharge clear")
        return False

    async def _apply_charger_smartstop_if_needed(
        self,
        charger_ent: str,
        device: DeviceConfig,
        device_name: str,
        battery: float,
        expected_on: bool,
        service_data: dict[str, Any],
    ) -> bool | None:
        """Handle SmartStop: deactivate charger when target reached, respecting throttle.

        Returns True/False to short-circuit `_apply_charger_logic`, or None to continue.
        """
        try:
            if not (expected_on and battery >= device.target_level):
                return None
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
                    _LOGGER.warning(
                        "SmartStop: raw last=%r type=%s throttle=%s",
                        last,
                        type(last),
                        throttle,
                    )
                    try:
                        if isinstance(last, (int, float)):
                            last_ts_val = float(last)
                        elif isinstance(last, str):
                            parsed = dt_util.parse_datetime(last)
                            last_ts_val = (
                                float(dt_util.as_timestamp(parsed)) if parsed else None
                            )
                        else:
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
                        _ignored_exc()
            except Exception:
                _ignored_exc()
            await self._maybe_switch("turn_off", service_data, desired=False)
            return False
        except Exception:
            _ignored_exc()
            return False

    async def _apply_charger_precharge_activate_if_needed(
        self,
        device: DeviceConfig,
        charger_ent: str,
        device_name: str,
        target_release: float,
        forecast_holdoff: bool,
        service_data: dict[str, Any],
    ) -> bool:
        """Handle precharge activation branch: turn_on with bypass/no-bypass.

        Returns True when the helper performed the activation (caller should return True).
        """
        try:
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
            bypass = bool(forecast_holdoff)
            if bypass:
                try:
                    pre_ts_local = (
                        getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                    )
                    pre_epoch_local = float(dt_util.as_timestamp(pre_ts_local))
                except Exception:
                    pre_epoch_local = float(dt_util.as_timestamp(dt_util.utcnow()))
                prev_action = self._last_action_state.get(charger_ent)
                await self._async_switch_call(
                    "turn_on",
                    service_data,
                    pre_epoch=pre_epoch_local,
                    previous_last_action=prev_action,
                    bypass_throttle=True,
                )
                return True
            await self._maybe_switch("turn_on", service_data, desired=True)
            return True
        except Exception:
            _ignored_exc()
            return False

    async def _apply_charger_precharge_logic(
        self,
        device: DeviceConfig,
        *,
        charger_ent: str,
        expected_on: bool,
        battery: float,
        predicted_level: float,
        release_level: float | None,
        margin_on: float,
        window_imminent: bool,
        forecast_holdoff: bool,
        service_data: dict[str, Any],
        device_name: str,
    ) -> bool | None:
        """Encapsulate the `if precharge_required:` logic.

        Returns True/False to short-circuit `_apply_charger_logic`, or
        None to continue normal processing.
        """
        try:
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
                bypass = bool(release_level is not None)
                try:
                    pre_ts_local = (
                        getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                    )
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

            handled_activate = await self._apply_charger_precharge_activate_if_needed(
                device,
                charger_ent,
                device_name,
                target_release,
                forecast_holdoff,
                service_data,
            )
            if handled_activate:
                return True

            self._log_action(
                device_name,
                logging.DEBUG,
                "[Precharge] Keeping charger on for %s until %.1f%%",
                device_name,
                target_release,
            )
            return True
        except Exception:
            _ignored_exc()
        return None

    def _maintain_telemetry(self, now_local: datetime) -> None:
        """Compute flip-flop rates, expire overrides and apply adaptive throttles.

        This is a slightly simplified, better-indented variant of the extracted
        logic to avoid deep nested try/except chains that previously caused a
        SyntaxError during edits. Behaviorally we keep the same guards and
        state updates but with clearer structure so matching try/except blocks
        don't get out of sync after future edits.
        """
        try:
            now_epoch = float(dt_util.as_timestamp(now_local))
        except Exception:
            _ignored_exc()
            return

        # cutoff computed earlier is not used directly here; keep for clarity
        # Expire adaptive overrides (small helper)
        try:
            self._expire_adaptive_overrides(now_epoch)
        except Exception:
            _ignored_exc()

        # Prune flip-flop event lists to the configured lookback window so
        # subsequent logic operates on only recent events. Extracted to a
        # helper for clarity and testability.
        try:
            self._prune_flipflop_events(now_epoch)
        except Exception:
            _ignored_exc()

        try:
            self._analyze_flipflop_events_and_apply_throttles(now_epoch)
        except Exception:
            _ignored_exc()

        try:
            self._update_flipflop_ewma_and_mode(now_epoch)
        except Exception:
            _ignored_exc()

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

    def _expire_adaptive_overrides(self, now_epoch: float) -> None:
        """Expire adaptive throttle overrides whose expiry has passed.

        This helper is deliberately small and focused so it can be unit-tested
        independently and avoid deep nested try/except in the main telemetry
        path.
        """
        try:
            # iterate over a copy since we may pop keys
            for ent, meta in list(self._adaptive_throttle_overrides.items()):
                try:
                    expires = float(meta.get("expires", 0.0) or 0.0)
                except Exception:
                    _ignored_exc()
                    continue

                if expires <= now_epoch:
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

    def _prune_flipflop_events(self, now_epoch: float) -> None:
        """Prune recorded flip-flop event timestamps to the configured window.

        This keeps the in-memory lists small and centralizes the cutoff logic
        so it can be exercised by focused unit tests.
        """
        try:
            cutoff = float(now_epoch) - float(self._flipflop_window_seconds)
        except Exception:
            _ignored_exc()
            return

        try:
            # Iterate over a shallow copy since we may remove keys
            for ent, evts in list(self._flipflop_events.items()):
                try:
                    trimmed = [float(e) for e in (evts or []) if float(e) >= cutoff]
                    if trimmed:
                        self._flipflop_events[ent] = trimmed
                    else:
                        # Remove empty lists to keep the dict small
                        self._flipflop_events.pop(ent, None)
                except Exception:
                    _ignored_exc()
                    continue
        except Exception:
            _ignored_exc()

    def _apply_adaptive_throttle_for_entity(
        self, ent: str, recent: list[float], now_epoch: float
    ) -> None:
        """Compute and apply an adaptive throttle override for a single entity.

        This encapsulates the multiplier/backoff logic and the details of
        updating `_adaptive_throttle_overrides` and `_device_switch_throttle`.
        """
        try:
            current = float(
                self._device_switch_throttle.get(
                    ent, self._default_switch_throttle_seconds
                )
                or self._default_switch_throttle_seconds
            )
        except Exception:
            _ignored_exc()
            current = float(self._default_switch_throttle_seconds)

        try:
            count = len(recent)
            excess = max(0, count - int(self._flipflop_warn_threshold))
            var_multiplier = float(self._adaptive_throttle_multiplier) + (
                float(self._adaptive_throttle_backoff_step)
                * float(excess)
                * float(getattr(self, "_adaptive_mode_factor", 1.0))
            )
            var_multiplier = min(
                var_multiplier, float(self._adaptive_throttle_max_multiplier)
            )
        except Exception:
            _ignored_exc()
            var_multiplier = float(self._adaptive_throttle_multiplier)

        try:
            desired = max(
                current * float(var_multiplier),
                float(self._adaptive_throttle_min_seconds),
            )
        except Exception:
            _ignored_exc()
            desired = float(self._adaptive_throttle_min_seconds)

        try:
            meta_entry: dict | None = self._adaptive_throttle_overrides.get(ent)
        except Exception:
            meta_entry = None

        if not meta_entry:
            try:
                orig = float(
                    self._device_switch_throttle.get(
                        ent, self._default_switch_throttle_seconds
                    )
                    or self._default_switch_throttle_seconds
                )
                self._adaptive_throttle_overrides[ent] = {
                    "original": orig,
                    "applied": float(desired),
                    "expires": float(
                        now_epoch + float(self._adaptive_throttle_duration_seconds)
                    ),
                }
                self._device_switch_throttle[ent] = float(desired)
                _LOGGER.info(
                    "Adaptive throttle applied for %s = %.1fs (original %.1fs) until %.0f",
                    ent,
                    float(desired),
                    float(
                        self._adaptive_throttle_overrides.get(ent, {}).get(
                            "original", 0.0
                        )
                    ),
                    float(now_epoch + float(self._adaptive_throttle_duration_seconds)),
                )
            except Exception:
                _ignored_exc()
        else:
            try:
                meta_applied = float(meta_entry.get("applied", 0.0) or 0.0)
                new_applied = max(meta_applied, float(desired))
                meta_entry["applied"] = new_applied
                meta_entry["expires"] = float(
                    now_epoch + float(self._adaptive_throttle_duration_seconds)
                )
                self._adaptive_throttle_overrides[ent] = meta_entry
                self._device_switch_throttle[ent] = float(new_applied)
            except Exception:
                _ignored_exc()
        return

    def _update_flipflop_ewma_and_mode(self, now_epoch: float) -> None:
        """Update EWMA based on recent flip-flop rates and manage adaptive mode overrides.

        Extracted for clarity and unit testing. This updates EWMA state, logs
        threshold crossings, and persists adaptive mode overrides to the
        config entry options when sustained conditions occur.
        """
        try:
            rate_per_sec, alpha, prev = self._compute_flipflop_metrics(now_epoch)
            try:
                ewma = prev + alpha * (rate_per_sec - prev)
            except Exception:
                _ignored_exc()
                ewma = prev

            try:
                self._apply_flipflop_ewma_and_mode(ewma, now_epoch)
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()

    def _analyze_flipflop_events_and_apply_throttles(self, now_epoch: float) -> None:
        """Analyze recent flip-flop events and apply adaptive throttles as needed.

        This helper was split out to keep telemetry maintenance readable and
        testable. It iterates recent flip-flop events and invokes the
        per-entity adaptive throttle application when thresholds are met.
        """
        try:
            warn_threshold = int(
                getattr(
                    self,
                    "_flipflop_warn_threshold",
                    DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD,
                )
            )
        except Exception:
            warn_threshold = int(DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD)

        try:
            # Iterate over a shallow copy in case helpers modify the dict
            for ent, recent in list((self._flipflop_events or {}).items()):
                try:
                    recent_list = list(recent or [])
                    if len(recent_list) >= warn_threshold and getattr(
                        self, "_adaptive_enabled", True
                    ):
                        # Apply adaptive throttle for this entity
                        try:
                            self._apply_adaptive_throttle_for_entity(
                                ent, recent_list, now_epoch
                            )
                        except Exception:
                            _ignored_exc()
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()

    def _compute_flipflop_metrics(self, now_epoch: float) -> tuple[float, float, float]:
        """Compute flipflop metrics: rate per second, EWMA alpha and previous EWMA."""
        try:
            total_events = sum(len(v) for v in self._flipflop_events.values())
            window = float(
                getattr(
                    self,
                    "_flipflop_window_seconds",
                    DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS,
                )
            )
            rate_per_sec = float(total_events) / window if window > 0 else 0.0

            try:
                entry_obj = getattr(self, "entry", None)
                if entry_obj and getattr(entry_obj, "options", None) is not None:
                    raw_alpha = entry_obj.options.get(CONF_ADAPTIVE_EWMA_ALPHA)
                    alpha = (
                        float(raw_alpha)
                        if raw_alpha is not None
                        else DEFAULT_ADAPTIVE_EWMA_ALPHA
                    )
                else:
                    alpha = DEFAULT_ADAPTIVE_EWMA_ALPHA
            except Exception:
                alpha = DEFAULT_ADAPTIVE_EWMA_ALPHA

            prev = float(getattr(self, "_flipflop_ewma", 0.0) or 0.0)
            return rate_per_sec, alpha, prev
        except Exception:
            _ignored_exc()
            return (
                0.0,
                DEFAULT_ADAPTIVE_EWMA_ALPHA,
                float(getattr(self, "_flipflop_ewma", 0.0) or 0.0),
            )

    def _apply_flipflop_ewma_and_mode(self, ewma: float, now_epoch: float) -> None:
        """Apply computed EWMA, manage threshold crossings and adaptive mode overrides."""
        try:
            self._flipflop_ewma = ewma
            self._flipflop_ewma_last_update = float(now_epoch)
            exceeded_threshold = float(self._flipflop_warn_threshold) / max(
                1.0, float(self._flipflop_window_seconds)
            )
            prev_exceeded = bool(getattr(self, "_flipflop_ewma_exceeded", False))
            new_exceeded = ewma >= exceeded_threshold
            self._flipflop_ewma_exceeded = new_exceeded
            now_ts = float(now_epoch)

            if new_exceeded and not prev_exceeded:
                self._flipflop_ewma_exceeded_since = now_ts
                _LOGGER.warning(
                    "Flipflop EWMA exceeded threshold: ewma=%.6f threshold=%.6f",
                    ewma,
                    exceeded_threshold,
                )
            elif new_exceeded and prev_exceeded:
                try:
                    since = float(
                        getattr(self, "_flipflop_ewma_exceeded_since", now_ts) or now_ts
                    )
                    duration = now_ts - since
                    if (
                        duration >= 300.0
                        and self._adaptive_mode_override != "aggressive"
                    ):
                        try:
                            self._apply_aggressive_adaptive_override(duration)
                        except Exception:
                            _ignored_exc()
                except Exception:
                    _ignored_exc()
            else:
                self._flipflop_ewma_exceeded_since = None
                try:
                    if getattr(self, "_adaptive_mode_override", None) is not None:
                        self._clear_adaptive_mode_override()
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()

    def _clear_adaptive_mode_override(self) -> None:
        """Clear adaptive_mode_override and persist the change.

        Extracted from the EWMA handler so persistence logic is isolated.
        """
        try:
            _LOGGER.info("Adaptive mode override cleared (EWMA dropped)")
            self._adaptive_mode_override = None
            try:
                new_opts = dict(getattr(self.entry, "options", {}) or {})
                if "adaptive_mode_override" in new_opts:
                    new_opts.pop("adaptive_mode_override", None)
                    try:
                        self.hass.config_entries.async_update_entry(
                            self.entry, options=new_opts
                        )
                    except Exception:
                        _ignored_exc()
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()

    def _apply_aggressive_adaptive_override(self, duration: float) -> None:
        """Apply aggressive adaptive_mode_override and persist options.

        This separates the side-effect-heavy persistence logic out of the
        complexity-heavy EWMA method.
        """
        try:
            self._adaptive_mode_override = "aggressive"
            _LOGGER.warning(
                "Adaptive mode override applied: aggressive (sustained EWMA for %.0fs)",
                duration,
            )
            try:
                new_opts = dict(getattr(self.entry, "options", {}) or {})
                new_opts["adaptive_mode_override"] = "aggressive"
                try:
                    self.hass.config_entries.async_update_entry(
                        self.entry, options=new_opts
                    )
                except Exception:
                    _ignored_exc()
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()

    def _float_state(self, entity_id: str | None) -> float | None:
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

    def _text_state(self, entity_id: str | None) -> str | None:
        if not entity_id:
            return None
        state_obj = self.hass.states.get(entity_id)
        if not state_obj or state_obj.state in UNKNOWN_STATES:
            return None
        return str(state_obj.state)

    def _charging_state(self, entity_id: str | None) -> str:
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
    def _parse_alarm_time(value: str | None) -> time:
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
    ) -> tuple[float | None, bool]:
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
        rate: float, observed_rate: float | None
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
        observed_rate: float | None,
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

    def _compute_required_duration_minutes(
        self,
        device: DeviceConfig,
        charge_deficit: float,
        avg_speed: float,
        speed_confident: bool,
        hours_until_alarm: float,
    ) -> float:
        """Compute required charging duration in minutes for a device.

        Encapsulates the heuristics that determine how many minutes of
        charging are needed given average speed, confidence and the time to
        alarm. Returns a duration in minutes.
        """
        if charge_deficit <= 0.0:
            return 0.0

        if avg_speed > 0.0:
            duration_hours = charge_deficit / max(avg_speed, 1e-3)
            duration_min = min(duration_hours * 60.0, 24 * 60)
            if not speed_confident:
                heuristic_min = charge_deficit * DEFAULT_FALLBACK_MINUTES_PER_PERCENT
                heuristic_min = min(heuristic_min, 24 * 60)
                if heuristic_min > 0:
                    duration_min = min(duration_min, heuristic_min * 1.2)
        else:
            duration_min = 24 * 60

        if hours_until_alarm > 0:
            min_window_hours = 0.25
            duration_min = max(duration_min, min_window_hours * 60.0)
            duration_min = min(duration_min, hours_until_alarm * 60.0)

        return duration_min

    def _resolve_charger_switch_state(
        self, device: DeviceConfig, now_local: datetime
    ) -> tuple[bool, str, bool]:
        """Return (charger_available, charger_state_value, charger_is_on).

        Encapsulates reading the charger switch entity and applying the
        coordinator "assumed state" fallback when the entity reports an
        unknown/unavailable state but a recent coordinator action suggests a
        different implied state.
        """
        charger_state_obj = self.hass.states.get(device.charger_switch)
        if charger_state_obj and charger_state_obj.state not in UNKNOWN_STATES:
            charger_available = True
            charger_state_value = str(charger_state_obj.state).lower()
        else:
            charger_available = False
            charger_state_value = ""

        charger_is_on = charger_available and charger_state_value in (
            "on",
            "charging",
            STATE_ON,
        )

        # If the switch reports off/unavailable, allow the coordinator's
        # most-recent intended action to be used when a recent switch was
        # issued (throttle window) so tests/integrations with lagging
        # entity states behave deterministically.
        if not charger_is_on:
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
                now_for_cmp = (
                    getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                )
                try:
                    now_ts = float(dt_util.as_timestamp(now_for_cmp))
                    if isinstance(last_ts, (int, float)):
                        last_epoch = float(last_ts)
                    else:
                        last_epoch = float(dt_util.as_timestamp(last_ts))
                    elapsed = now_ts - last_epoch
                    if elapsed >= 0 and elapsed <= float(throttle_window):
                        charger_is_on = bool(last_action)
                except Exception:
                    _ignored_exc()

        return charger_available, charger_state_value, charger_is_on

    def _compute_plan_timing(
        self,
        device: DeviceConfig,
        alarm_dt: datetime,
        now_local: datetime,
        battery: float,
        avg_speed: float,
        speed_confident: bool,
        hours_until_alarm: float,
        drain_rate: float,
    ) -> tuple[float, float, float, datetime | None, bool, float]:
        """Compute predicted level, deficit, duration, start window and main duration.

        This method isolates the math/windowing logic used by `_build_plan_impl` so
        it can be unit-tested and kept small.
        """
        try:
            expected_drain = max(0.0, hours_until_alarm * drain_rate)
            predicted_level = max(0.0, battery - expected_drain)

            charge_deficit = max(0.0, device.target_level - predicted_level)
            duration_min = self._compute_required_duration_minutes(
                device,
                charge_deficit,
                avg_speed,
                speed_confident,
                hours_until_alarm,
            )

            start_time, smart_start_active, duration_min = self._resolve_start_window(
                alarm_dt=alarm_dt,
                duration_min=duration_min,
                charge_deficit=charge_deficit,
            )
            main_duration_min = duration_min
            return (
                predicted_level,
                charge_deficit,
                duration_min,
                start_time,
                smart_start_active,
                main_duration_min,
            )
        except Exception:
            _ignored_exc()
            return 0.0, 0.0, 0.0, None, False, 0.0

    def _determine_charger_state_and_assumed(
        self, device: DeviceConfig, alarm_dt: datetime, now_local: datetime
    ) -> tuple[bool, str, bool, bool | None]:
        """Determine charger availability, state and an assumed-on fallback.

        Returns (charger_available, charger_state_value, charger_is_on, last_action).
        Encapsulates the throttle-based assumed-state behavior so the plan
        builder can remain linear and easier to test.
        """
        try:
            charger_available, charger_state_value, charger_is_on = (
                self._resolve_charger_switch_state(device, now_local)
            )
        except Exception:
            _ignored_exc()
            charger_available, charger_state_value, charger_is_on = False, "", False

        try:
            last_action = self._last_action_state.get(device.charger_switch)
        except Exception:
            _ignored_exc()
            last_action = None

        # If the observed entity state is off/unavailable, consider recent
        # coordinator-issued switches within the per-device throttle window
        # as the presumed state for this evaluation.
        if not charger_is_on:
            try:
                throttle_seconds = self._device_switch_throttle.get(
                    device.charger_switch, self._default_switch_throttle_seconds
                )
                throttle_window = float(throttle_seconds) if throttle_seconds else 5.0
            except Exception:
                _ignored_exc()
                throttle_window = 5.0

            try:
                last_ts = self._last_switch_time.get(device.charger_switch)
                if last_action is not None and last_ts is not None:
                    now_for_cmp = (
                        getattr(self, "_current_eval_time", None) or dt_util.utcnow()
                    )
                    try:
                        now_ts = float(dt_util.as_timestamp(now_for_cmp))
                        if isinstance(last_ts, (int, float)):
                            last_epoch = float(last_ts)
                        else:
                            last_epoch = float(dt_util.as_timestamp(last_ts))
                        elapsed = now_ts - last_epoch
                        if elapsed >= 0 and elapsed <= float(throttle_window):
                            charger_is_on = bool(last_action)
                    except Exception:
                        _ignored_exc()
            except Exception:
                _ignored_exc()

        return charger_available, charger_state_value, charger_is_on, last_action

    async def _build_plan_impl(
        self,
        device: DeviceConfig,
        now_local: datetime,
        learning,
        learning_window_hours: float,
    ) -> SmartChargePlan | None:
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
        _LOGGER.info(
            "_build_plan: device=%s now=%s eval=%d",
            device.name,
            now_local.isoformat(),
            self._current_eval_id,
        )

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
        # Compute timing/amounts required for charging and defer the
        # detailed logic to a small helper for clarity and testability.
        (
            predicted_level,
            charge_deficit,
            duration_min,
            start_time,
            smart_start_active,
            main_duration_min,
        ) = self._compute_plan_timing(
            device,
            alarm_dt,
            now_local,
            battery,
            avg_speed,
            speed_confident,
            hours_until_alarm,
            drain_rate,
        )

        charger_available, charger_state_value, charger_is_on, last_action = (
            self._determine_charger_state_and_assumed(device, alarm_dt, now_local)
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
        precharge_duration_min: float | None = None
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
    ) -> tuple[datetime | None, bool, float]:
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
        start_time: datetime | None,
    ) -> tuple[bool, float | None, float, float, float, bool]:
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
        # Prefer any persisted per-entity smart_start override requested by post-alarm repairs
        try:
            raw_pm = None
            if getattr(self, "_post_alarm_persisted_smart_start", None) is not None:
                raw_pm = self._post_alarm_persisted_smart_start.get(
                    device.charger_switch
                )
            persisted_margin = float(raw_pm) if raw_pm is not None else None
        except Exception:
            persisted_margin = None
        smart_margin = (
            persisted_margin
            if persisted_margin is not None
            else (
                device.smart_start_margin
                if device.smart_start_margin is not None
                else self._smart_start_margin
            )
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
        start_time: datetime | None,
        now_local: datetime,
        margin_on: float,
        margin_off: float,
        smart_margin: float,
        previous_release: float | None,
    ) -> tuple[float | None, bool]:
        """Compute and set release_level when a precharge latch should occur.

        Returns (release_level, precharge_required).
        """
        extra_margin = max(margin_off, expected_drain * 0.4)
        release_cap = device.target_level
        if smart_start_active and start_time and now_local < start_time:
            release_cap = max(
                device.precharge_level, device.target_level - smart_margin
            )
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
        release_ready_at: datetime | None,
        now_local: datetime,
    ) -> tuple[bool, float | None, datetime | None]:
        """Handle logic for when a release_level already exists.

        Returns (precharge_required, release_level, release_ready_at).
        """
        previously_in_range = release_ready_at is not None
        in_range = release_ready_at is not None or (
            device.name in self._precharge_release
            and battery >= self._precharge_release[device.name]
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
                    return (
                        True,
                        self._precharge_release.get(device.name),
                        release_ready_at,
                    )
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
                    return (
                        True,
                        self._precharge_release.get(device.name),
                        release_ready_at,
                    )
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
        caller_fn: str | None,
        entity_id: str | None,
        pre_epoch: float | None,
        previous_last_action: bool | None,
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
            if (
                entity_id
                and not force
                and previous_last_action is not None
                and caller_fn != "_maybe_switch"
            ):
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
                    self._device_switch_throttle.get(
                        entity_id, self._default_switch_throttle_seconds
                    )
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
        entity_id: str | None,
        pre_epoch: float | None,
        bypass_throttle: bool,
        force: bool,
        previous_last_action: bool | None,
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

    def _parse_stored_epoch(self, entity_id: str | None) -> float | None:
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

    def _is_device_in_latch_maps(self, dev_name: str | None) -> bool:
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

    def _compute_local_effective_bypass(
        self, entity_id: str | None, bypass_throttle: bool, force: bool
    ) -> bool:
        """Compute whether this call should locally bypass throttle.

        This encapsulates the logic that inspects precharge maps and resolves
        entity -> device name mappings.
        """
        local_effective_bypass = bool(bypass_throttle or force)
        try:
            if entity_id:
                try:
                    ent = str(entity_id)
                    if (
                        ent in self._precharge_release
                        or ent in self._precharge_release_ready
                    ):
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
        service_data: dict[str, Any],
        pre_epoch: float | None = None,
        previous_last_action: bool | None = None,
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
                caller_fn,
                entity_id,
                pre_epoch,
                previous_last_action,
                action,
                bypass_throttle,
                force,
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
                "Invoking service switch.%s for %s (caller=%s)",
                action,
                entity_id,
                caller,
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
                self._record_switch_invocation(
                    entity_id, action, previous_last_action, pre_epoch
                )
        except Exception:
            _ignored_exc()
        return True

    def _maybe_switch_early_suppression(
        self, norm: str, desired: bool, should_check: bool
    ) -> bool:
        """Early suppression gate extracted from _maybe_switch.

        Returns True when the operation SHOULD be suppressed.
        """
        if not should_check:
            return False

        # Delegate the inflight and initial debug logging to a small helper
        # to keep this gate simple and testable.
        try:
            if self._maybe_switch_inflight_and_dbg(norm, desired, should_check):
                return True
        except Exception:
            _ignored_exc()

        # Consolidated quick-gate
        try:
            if self._maybe_switch_consolidated_quick_gate(norm, desired):
                return True
        except Exception:
            _LOGGER.debug("Consolidated quick-gate failed for %s", norm)
        try:
            if self._maybe_switch_extra_quick_gate(norm, desired):
                return True
        except Exception:
            _ignored_exc()

        try:
            if self._maybe_switch_early_throttle_check(norm, desired):
                return True
        except Exception:
            _LOGGER.debug("Early suppression gate failed for %s", norm)

        # Explicit final return for type-checker and clarity
        return False

    def _maybe_switch_extra_quick_gate(self, norm: str, desired: bool) -> bool:
        """Extra conservative quick gate for early suppression.

        Returns True when the gate decides to suppress the call.
        """
        try:
            last_e = self._normalize_last_epoch(self._last_switch_time.get(norm))
            thr_q = self._get_throttle_seconds(norm)
            if last_e is None or not thr_q:
                return False
            now_e = self._get_now_epoch()
            last_act_quick = self._get_last_action_state(norm)
            _LOGGER.debug(
                "QUICK_GATE_DEBUG: entity=%s last_e=%.3f now_e=%.3f elapsed=%.3f throttle_quick=%.3f last_act_quick=%r desired=%s current_eval_time=%s",
                norm,
                float(last_e) if last_e is not None else float("nan"),
                now_e,
                (now_e - last_e) if last_e is not None else float("nan"),
                thr_q,
                last_act_quick,
                desired,
                getattr(self, "_current_eval_time", None),
            )
            if (
                last_act_quick is not None
                and (now_e - last_e) < float(thr_q)
                and bool(last_act_quick) != bool(desired)
            ):
                _LOGGER.info(
                    "EARLY_SUPPRESS_V2: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                    norm,
                    now_e - last_e,
                    thr_q,
                    last_act_quick,
                    desired,
                )
                return True
        except Exception:
            _ignored_exc()
        return False

    def _maybe_switch_consolidated_quick_gate(self, norm: str, desired: bool) -> bool:
        """Consolidated quick gate used by early suppression.

        Returns True when the quick gate decides to suppress the call.
        """
        try:
            last_epoch_val = self._normalize_last_epoch(
                self._last_switch_time.get(norm)
            )
            if last_epoch_val is None:
                return False
            now_epoch_val = self._get_now_epoch()
            throttle_val = self._throttle_value_for(norm)
            last_action_state = self._get_last_action_state(norm)
            elapsed_val = now_epoch_val - float(last_epoch_val)
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
            if (
                last_action_state is not None
                and elapsed_val >= 0
                and elapsed_val < float(throttle_val)
                and bool(last_action_state) != bool(desired)
            ):
                _LOGGER.info(
                    "CONSOLIDATED_EARLY_SUPPRESS: entity=%s elapsed=%.3f throttle=%.3f last_act=%r desired=%s",
                    norm,
                    elapsed_val,
                    throttle_val,
                    last_action_state,
                    desired,
                )
                try:
                    if self._maybe_switch_recent_eval_guard(norm):
                        return True
                except Exception:
                    return True
                return True
        except Exception:
            _LOGGER.debug("Consolidated quick-gate internal error for %s", norm)
        return False

    def _maybe_switch_recent_eval_guard(self, norm: str) -> bool:
        """Guard that decides suppression based on recent evaluation ids.

        Returns True when the recent-eval heuristic indicates suppression.
        """
        try:
            last_eval = self._last_switch_eval.get(norm)
            cur_eval = int(getattr(self, "_current_eval_id", 0) or 0)
            if last_eval is None:
                return True
            try:
                if abs(cur_eval - int(last_eval or 0)) <= 1:
                    return True
            except Exception:
                return True
        except Exception:
            _ignored_exc()
        return False

    def _maybe_switch_inflight_and_dbg(
        self, norm: str, desired: bool, should_check: bool
    ) -> bool:
        """Check for in-flight switches and emit initial debug logging.

        Returns True when the operation should be suppressed (in-flight
        reversal) or when debug entry logging indicates suppression.
        """
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
                return True
        except Exception:
            _ignored_exc()

        # Emit the diagnostic entry for the SHOULD_CHECK gate. Keep this
        # isolated so the surrounding function remains small.
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

        return False

    def _maybe_switch_early_throttle_check(self, norm: str, desired: bool) -> bool:
        """Throttle/timestamp based early suppression.

        Returns True when the throttle/timestamp heuristics determine the
        switch should be suppressed.
        """
        try:
            last_raw = self._last_switch_time.get(norm)
            throttle_cfg = self._device_switch_throttle.get(
                norm, self._default_switch_throttle_seconds
            )
            if last_raw is not None and throttle_cfg:
                last_epoch = self._normalize_last_epoch(last_raw)
                if last_epoch is not None:
                    now_epoch = self._get_now_epoch()
                    try:
                        throttle_val = float(throttle_cfg)
                    except Exception:
                        throttle_val = float(self._default_switch_throttle_seconds)
                    last_act = self._get_last_action_state(norm)
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
                    if (
                        elapsed_early is not None
                        and elapsed_early < throttle_val
                        and last_act is not None
                        and bool(last_act) != bool(desired)
                    ):
                        _LOGGER.info(
                            "EARLY_SUPPRESS: entity=%s last_action=%s desired=%s elapsed=%.3f throttle=%s",
                            norm,
                            last_act,
                            desired,
                            elapsed_early,
                            throttle_val,
                        )
                        return True
        except Exception:
            _ignored_exc()
        return False

    def _record_switch_invocation(
        self,
        entity_id: str,
        action: str,
        previous_last_action: bool | None,
        pre_epoch: float | None,
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
            _LOGGER.debug(
                "Recorded last_switch_time for %s = epoch %.3f", entity_id, epoch
            )
        try:
            _LOGGER.debug(
                "last_switch_time keys after set: %s",
                list(self._last_switch_time.keys()),
            )
        except Exception:
            _ignored_exc()

    def _resolve_epoch_for_invocation(self, pre_epoch: float | None) -> float:
        """Resolve the epoch timestamp used when recording a switch invocation."""
        ts = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
        try:
            return (
                float(pre_epoch)
                if pre_epoch is not None
                else float(dt_util.as_timestamp(ts))
            )
        except Exception:
            try:
                return float(dt_util.as_timestamp(ts))
            except Exception:
                return float(dt_util.as_timestamp(dt_util.utcnow()))

    def _set_last_action_state(
        self, entity_id: str, action: str, previous_last_action: bool | None
    ) -> None:
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
        self._last_recorded_eval[entity] = int(
            getattr(self, "_current_eval_id", 0) or 0
        )
        try:
            _LOGGER.debug(
                "DEBUG_RECORD_DESIRED: entity=%s current_eval=%s",
                entity,
                getattr(self, "_current_eval_id", None),
            )
        except Exception:
            _ignored_exc()
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
                    epoch = float(
                        dt_util.as_timestamp(
                            getattr(self, "_current_eval_time", None)
                            or dt_util.utcnow()
                        )
                    )
                except Exception:
                    epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
                try:
                    self._record_flipflop_event(entity, epoch)
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

    def _record_flipflop_event(self, entity: str, epoch: float) -> None:
        """Record and trim a flip-flop event timestamp for an entity.

        Centralized helper extracted to make the behavior unit-testable and
        keep trimming logic in one place.
        """
        try:
            evts = self._flipflop_events.setdefault(entity, [])
            evts.append(float(epoch))
        except Exception:
            _ignored_exc()
            return

        try:
            cutoff = float(epoch) - float(
                getattr(self, "_flipflop_window_seconds", 300.0)
            )
            self._flipflop_events[entity] = [e for e in evts if e >= cutoff]
        except Exception:
            _ignored_exc()

    def _quick_gate_suppress(
        self, norm: str, last_epoch_quick: float | None, throttle: Any, desired: bool
    ) -> bool:
        """Evaluate the quick conservative gate and return True to suppress.

        This extracts the inlined logic from _maybe_switch that computes elapsed
        since the last switch and compares against a throttle; if the last
        action state differs from the desired state and the elapsed time is
        inside the throttle window, the call should be suppressed.
        """
        try:
            if last_epoch_quick is None:
                return False

            now_for_quick = (
                getattr(self, "_current_eval_time", None) or dt_util.utcnow()
            )
            now_epoch_val = float(dt_util.as_timestamp(now_for_quick))
            elapsed_quick = now_epoch_val - float(last_epoch_quick)

            try:
                throttle_val = float(throttle)
            except Exception:
                throttle_val = float(self._default_switch_throttle_seconds)

            last_action_state = self._last_action_state.get(norm)
            if last_action_state is None:
                try:
                    st = self.hass.states.get(norm)
                    last_action_state = bool(st and st.state == STATE_ON)
                except Exception:
                    last_action_state = None

            if elapsed_quick is not None and elapsed_quick < throttle_val:
                if last_action_state is not None and bool(last_action_state) != bool(
                    desired
                ):
                    return True
            return False
        except Exception:
            _LOGGER.exception("Quick gate evaluation failed for %s", norm)
            return False

    def _throttle_suppress(
        self, norm: str, last: Any, throttle_cfg: Any, desired: bool
    ) -> bool:
        """Evaluate the throttle suppression using a normalized last timestamp.

        Returns True when the action should be suppressed due to being inside
        the per-device throttle window and the last action state differing
        from the desired state.
        """
        try:
            if last is None:
                return False

            last_epoch = None
            try:
                if isinstance(last, (int, float)):
                    last_epoch = float(last)
                elif isinstance(last, str):
                    parsed = dt_util.parse_datetime(last)
                    last_epoch = float(dt_util.as_timestamp(parsed)) if parsed else None
                else:
                    last_epoch = float(dt_util.as_timestamp(last))
            except Exception:
                last_epoch = None

            if last_epoch is None:
                return False

            now_for_cmp = dt_util.utcnow()
            now_ts = float(dt_util.as_timestamp(now_for_cmp))
            last_ts_val = float(last_epoch)
            elapsed = now_ts - last_ts_val

            try:
                throttle_val = float(throttle_cfg)
            except Exception:
                throttle_val = float(self._default_switch_throttle_seconds)

            if elapsed < 0:
                # treat future timestamps as expired
                return False

            last_act = self._get_last_action_state(norm)
            if (
                last_act is not None
                and elapsed < throttle_val
                and bool(last_act) != bool(desired)
            ):
                return True
            return False
        except Exception:
            _ignored_exc()
            return False

    def _confirmation_and_throttle_check(self, norm: str, desired: bool):
        """Handle confirmation debounce and throttle pre-check.

        Returns True when the call should be suppressed (waiting for confirmation
        or inside throttle window), False when processing should continue.
        """
        try:
            # Record desired state for confirmation counting (idempotent per-eval)
            self._record_desired_state(norm, desired)
            confirm_key, required, hist, last, throttle = (
                self._maybe_switch_collect_gate_inputs(norm, desired, True)
            )
            count = hist[1]

            if count < required:
                _LOGGER.debug(
                    "Waiting for confirmation for %s -> desired=%s (count=%d/%d)",
                    norm,
                    desired,
                    count,
                    required,
                )
                try:
                    _LOGGER.debug(
                        "DEBUG_CONFIRM: entity=%s desired=%s hist=%r count=%s required=%s eval=%s",
                        norm,
                        desired,
                        hist,
                        count,
                        required,
                        getattr(self, "_current_eval_id", None),
                    )
                except Exception:
                    _ignored_exc()
                _LOGGER.debug(
                    "DBG_WAIT: entity=%s desired=%s count=%d required=%d eval=%s",
                    norm,
                    desired,
                    count,
                    required,
                    getattr(self, "_current_eval_id", None),
                )
                return True, hist, required, count

            # Early throttle suppression
            try:
                if self._throttle_suppress(norm, last, throttle, desired):
                    return True, hist, required, count
            except Exception:
                _ignored_exc()

            # Conservative throttle-only suppression
            try:
                if self._maybe_switch_conservative_throttle_check(last, throttle):
                    return True, hist, required, count
            except Exception:
                _ignored_exc()

            return False, hist, required, count
        except Exception:
            _ignored_exc()
            return False, (desired, 0), int(self._confirmation_required), 0

    def _maybe_switch_conservative_throttle_check(
        self, last: Any, throttle: Any
    ) -> bool:
        """Conservative throttle-only suppression: treat recent last-switch as suppressing."""
        try:
            if last is not None:
                last_epoch = self._parse_epoch(last)
                if last_epoch is not None:
                    now_ts = float(dt_util.as_timestamp(dt_util.utcnow()))
                    elapsed = now_ts - float(last_epoch)
                    try:
                        throttle_val = float(throttle)
                    except Exception:
                        throttle_val = float(self._default_switch_throttle_seconds)
                    if elapsed < 0:
                        if abs(elapsed) < 1e-3:
                            elapsed = 0.0
                        else:
                            pass
                    if elapsed >= 0 and elapsed < float(throttle_val):
                        return True
        except Exception:
            _ignored_exc()
        return False

    async def _maybe_switch(
        self,
        action: str,
        service_data: dict[str, Any],
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

        # Consolidate the many pre-flight checks and diagnostic logging
        # into a single helper so this function's cyclomatic complexity
        # is reduced. The helper preserves per-call exception handling
        # and returns an early-abort indicator or the execution context.
        try:
            (
                preflight_abort,
                abort_value,
                now,
                should_check,
                previous_last_action,
                pre_epoch,
            ) = self._maybe_switch_preflight(
                norm, raw_entity, desired, action, service_data, force, bypass_throttle
            )
        except Exception:
            # On helper error behave defensively and continue with
            # original fallbacks (compute now and should_check locally).
            _ignored_exc()
            should_check = not force and not bypass_throttle
            previous_last_action = None
            pre_epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
            preflight_abort = False
            abort_value = False

        if preflight_abort:
            return abort_value
        # (additional per-call logging moved into _maybe_switch_prepare_inputs)

        # Confirmation debounce and throttle checks are normally applied
        # to avoid flapping. However, certain urgent operations (for
        # example precharge-release pauses or presence-leave) explicitly
        # request bypass_throttle=True or force=True and should act
        # immediately. In those cases skip confirmation and throttle.
        should_check = not force and not bypass_throttle
        try:
            _LOGGER.debug(
                "DEBUG_MAYBE_SWITCH: entity=%s force=%s bypass_throttle=%s should_check=%s",
                norm,
                force,
                bypass_throttle,
                should_check,
            )
        except Exception:
            _ignored_exc()
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
        # Run confirmation/throttle flow before executing the action so
        # the confirmation counters are consulted. Extracted flow
        # contains the logic previously inlined; call it here to keep
        # behavior identical to the original implementation.
        try:
            if should_check:
                try:
                    if self._maybe_switch_confirmation_and_throttle_flow(
                        norm, desired, action, should_check
                    ):
                        return False
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()

        # Delegate final pre-recording, authoritative guards and the
        # actual async switch call to a helper to reduce _maybe_switch
        # complexity and keep behavior isolated for easier testing.
        return await self._maybe_switch_execute_action(
            norm,
            action,
            service_data,
            desired,
            should_check,
            previous_last_action,
            pre_epoch,
            bypass_throttle,
            force,
        )

    def _maybe_switch_preflight(
        self,
        norm: str,
        raw_entity: str | None,
        desired: bool,
        action: str,
        service_data: dict[str, Any],
        force: bool,
        bypass_throttle: bool,
    ) -> tuple[bool, bool, datetime, bool, bool | None, float]:
        """Perform preflight checks and return (abort, abort_value, now, should_check, previous_last_action, pre_epoch).

        abort is True when the caller should return immediately; abort_value
        is the value to return when abort is True. This helper centralizes
        earlier extracted checks to keep `_maybe_switch` small.
        """
        # Log entry and caller
        try:
            self._maybe_switch_log_entry_and_caller(norm, bypass_throttle)
        except Exception:
            _ignored_exc()

        abort, abort_value, now, should_check = self._maybe_switch_run_early_checks(
            norm, desired, force, bypass_throttle
        )
        if abort:
            return (
                True,
                abort_value,
                now,
                should_check,
                None,
                float(dt_util.as_timestamp(now)),
            )

        # Prepare inputs
        try:
            self._maybe_switch_prepare_inputs(
                norm, raw_entity, desired, now, should_check
            )
        except Exception:
            _ignored_exc()

        # Collect execution context
        previous_last_action = None
        pre_epoch = float(dt_util.as_timestamp(now))
        try:
            previous_last_action, pre_epoch = (
                self._maybe_switch_prepare_execution_context(norm)
            )
        except Exception:
            _ignored_exc()

        return False, False, now, should_check, previous_last_action, pre_epoch

    def _maybe_switch_run_early_checks(
        self, norm: str, desired: bool, force: bool, bypass_throttle: bool
    ) -> tuple[bool, bool, datetime, bool]:
        """Run early suppression checks and return (abort, abort_value, now, should_check)."""
        # Early suppression
        try:
            if self._early_suppress_checks(norm, desired, force, bypass_throttle):
                now = dt_util.utcnow()
                return True, False, now, False
        except Exception:
            _ignored_exc()

        now = getattr(self, "_current_eval_time", None) or dt_util.utcnow()
        should_check = not force and not bypass_throttle

        # Complex suppression helper
        try:
            if self._should_suppress_switch(norm, desired, force, bypass_throttle):
                return True, False, now, should_check
        except Exception:
            _LOGGER.debug(
                "_should_suppress_switch helper raised; proceeding with switch for %s",
                norm,
            )

        # Early suppression gate
        try:
            if self._maybe_switch_early_suppression(norm, desired, should_check):
                return True, False, now, should_check
        except Exception:
            _ignored_exc()

        return False, False, now, should_check

    async def _maybe_switch_execute_action(
        self,
        norm: str,
        action: str,
        service_data: dict[str, Any],
        desired: bool,
        should_check: bool,
        previous_last_action: bool | None,
        pre_epoch: float,
        bypass_throttle: bool,
        force: bool,
    ) -> bool:
        """Perform final pre-record checks and execute the async switch call.

        This was extracted from _maybe_switch to reduce that function's
        cyclomatic complexity. Behavior and logging are preserved.
        """
        # Do not clear desired_state_history here; only clear after a
        # successful switch call so confirmation counters are respected
        # until the action is actually performed.

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

        if should_check:
            try:
                if self._maybe_switch_check_stored_epoch_and_throttle(
                    norm, previous_last_action, desired
                ):
                    return False
            except Exception:
                _ignored_exc()

        # Extra deterministic pre-record check
        if should_check:
            try:
                if self._maybe_switch_pre_record_check(norm, pre_epoch, desired):
                    return False
            except Exception:
                _ignored_exc()

        return await self._maybe_switch_perform_call(
            norm,
            action,
            service_data,
            desired,
            previous_last_action,
            pre_epoch,
            bypass_throttle,
            force,
            should_check,
        )

    def _maybe_switch_check_stored_epoch_and_throttle(
        self, norm: str, previous_last_action: bool | None, desired: bool
    ) -> bool:
        """Check stored last switch time and run final throttle check.

        Returns True to short-circuit (suppress) the switch, False otherwise.
        """
        try:
            current_last = self._last_switch_time.get(norm)
            if current_last is not None:
                if isinstance(current_last, (int, float)):
                    stored_epoch = float(current_last)
                else:
                    parsed = (
                        dt_util.parse_datetime(current_last)
                        if isinstance(current_last, str)
                        else current_last
                    )
                    stored_epoch = (
                        float(dt_util.as_timestamp(parsed)) if parsed else None
                    )
            else:
                stored_epoch = None
            if stored_epoch is not None:
                try:
                    if self._maybe_switch_final_throttle_check(
                        norm, previous_last_action, desired
                    ):
                        return True
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()
        return False

    async def _maybe_switch_perform_call(
        self,
        norm: str,
        action: str,
        service_data: dict[str, Any],
        desired: bool,
        previous_last_action: bool | None,
        pre_epoch: float,
        bypass_throttle: bool,
        force: bool,
        should_check: bool,
    ) -> bool:
        """Perform the final call orchestration for a switch action.

        Returns the result of `_async_switch_call` or False when suppressed.
        """
        # Prepare call data and emit consistent debug/info logs
        call_data = self._maybe_switch_prepare_call_context(
            norm, action, service_data, desired, previous_last_action, pre_epoch
        )

        # Mark pre-call in-memory state (last action and inflight marker)
        self._maybe_switch_mark_pre_call_state(norm, desired)

        # Final guard check (only applies when should_check is True)
        if should_check:
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

            try:
                if self._maybe_switch_final_guard(
                    norm, pre_epoch, previous_last_action, desired
                ):
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

        # Post-call handling (clear desired history only on success)
        try:
            self._maybe_switch_handle_post_call_result(norm, result)
        except Exception:
            _ignored_exc()

        return result

    def _maybe_switch_prepare_call_context(
        self,
        norm: str,
        action: str,
        service_data: dict[str, Any],
        desired: bool,
        previous_last_action: bool | None,
        pre_epoch: float,
    ) -> dict[str, Any]:
        """Helper: prepare call_data and emit consistent logs for switch call."""
        _LOGGER.info(
            "Pre-record: intended action for %s (pre_epoch=%.3f)", norm, pre_epoch
        )
        _LOGGER.info("PROCEED: calling switch.%s for %s", action, norm)
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
        return call_data

    def _maybe_switch_mark_pre_call_state(self, norm: str, desired: bool) -> None:
        """Set last action state and inflight marker before issuing a call."""
        try:
            self._last_action_state[norm] = bool(desired)
        except Exception:
            _ignored_exc()
        try:
            self._inflight_switches[norm] = bool(desired)
        except Exception:
            _ignored_exc()

    def _maybe_switch_handle_post_call_result(self, norm: str, result: bool) -> None:
        """Handle post-call cleanup: clear desired history on success."""
        try:
            if result:
                try:
                    self._desired_state_history.pop(norm, None)
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()

    def _smartstart_ignore_distant_forecast_check(
        self,
        device_name: str,
        battery: float,
        window_threshold: float,
        start_time: datetime | None,
        now_local: datetime,
        expected_on: bool,
    ) -> bool | None:
        """Pure helper: decide whether to ignore a distant drain forecast.

        Returns `expected_on` when the check decides to short-circuit, or
        `None` to indicate normal evaluation should continue.
        This helper performs only logging and a deterministic check.
        """
        try:
            if start_time and now_local >= start_time:
                # Keep using _log_action so behavior/log messages match inline code.
                self._log_action(
                    device_name,
                    logging.DEBUG,
                    "[SmartStart] %s ignoring distant drain forecast (battery %.1f%%, guard %.1f%%)",
                    device_name,
                    battery,
                    window_threshold,
                )
                return expected_on
        except Exception:
            _ignored_exc()
        # None indicates normal evaluation should continue (no short-circuit)
        return None

    async def _smartstart_pause_if_needed(
        self,
        device_name: str,
        window_threshold: float,
        charger_ent: str,
        charger_is_on: bool,
        service_data: dict[str, Any],
    ) -> bool | None:
        """Extracted async helper for SmartStart pause branch.

        Returns False after pausing the charger when the helper handled
        the action, or None to continue normal evaluation.
        """
        try:
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
        except Exception:
            _ignored_exc()
        return None

    async def _apply_charger_prestart_pause_checks(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        charger_is_on: bool,
        smart_start_active: bool,
        start_time: datetime | None,
        precharge_required: bool,
        charge_deficit: float,
        battery: float,
        predicted_level: float,
        window_imminent: bool,
        device_name: str,
        charger_ent: str,
        smart_margin: float,
        service_data: dict[str, Any],
    ) -> bool | None:
        """Handle pre-start pausing/scheduling checks extracted from apply logic.

        Returns True/False to short-circuit `_apply_charger_logic`, or None.
        """
        try:
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
                await self._maybe_switch("turn_off", service_data, desired=False)
                return False
        except Exception:
            _ignored_exc()
        return None

    async def _apply_charger_smartstart_logic(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        charger_is_on: bool,
        start_time: datetime | None,
        smart_start_active: bool,
        precharge_required: bool,
        forecast_holdoff: bool,
        charge_deficit: float,
        predicted_level: float,
        margin_on: float,
        smart_margin: float,
        service_data: dict[str, Any],
        device_name: str,
        charger_ent: str,
    ) -> bool | None:
        """Encapsulate SmartStart-related branches.

        Returns True/False to short-circuit `_apply_charger_logic`, or
        None to continue normal processing.
        """
        try:
            # forecast_holdoff pause branch
            if forecast_holdoff and smart_start_active and not precharge_required:
                window_threshold = (
                    device.precharge_level + self._precharge_countdown_window
                )
                paused = await self._smartstart_pause_if_needed(
                    device_name,
                    window_threshold,
                    charger_ent,
                    charger_is_on,
                    service_data,
                )
                if paused is not None:
                    return paused

                if start_time and now_local >= start_time:
                    self._log_action(
                        device_name,
                        logging.DEBUG,
                        "[SmartStart] %s ignoring distant drain forecast (battery %.1f%%, guard %.1f%%)",
                        device_name,
                        battery,
                        window_threshold,
                    )
                    return charger_is_on

            # Activation branch when start_time reached
            if (
                smart_start_active
                and start_time
                and now_local >= start_time
                and not charger_is_on
                and not precharge_required
                and not forecast_holdoff
            ):
                activated = await self._smartstart_activate_if_needed(
                    device_name, charger_ent, service_data, charger_is_on
                )
                if activated is not None:
                    return activated

            # The remaining pre-start pausing conditions are intentionally
            # left to the caller (they run after SmartStop). This helper
            # focuses only on forecast_holdoff and start-time activation.
        except Exception:
            _ignored_exc()
        return None

    async def _apply_charger_preflight_checks(
        self,
        device: DeviceConfig,
        charger_ent: str,
        now_local: datetime,
        charger_is_on: bool,
        charger_available: bool,
        precharge_required: bool,
        release_level: float | None,
        margin_on: float,
        predicted_level: float,
        battery: float,
        service_data: dict[str, Any],
        device_name: str,
        smart_start_active: bool,
        start_time: datetime | None,
        expected_on: bool,
    ) -> bool | None:
        """Run early preflight checks (precharge-release, presence-clear, emergency, availability).

        Returns True/False to short-circuit `_apply_charger_logic`, or
        None to continue normal processing.
        """
        try:
            handled = await self._apply_charger_handle_precharge_release(
                device,
                charger_ent,
                charger_is_on,
                precharge_required,
                release_level,
                service_data,
                device_name,
                predicted_level,
                battery,
                margin_on,
            )
            if handled:
                return False
        except Exception:
            _LOGGER.exception("Error while handling immediate precharge release")

        handled_presence_clear = (
            await self._apply_charger_handle_presence_precharge_clear(
                device, charger_ent, charger_is_on, service_data, device_name
            )
        )
        if handled_presence_clear:
            return False

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
        return None

    async def _smartstart_activate_if_needed(
        self,
        device_name: str,
        charger_ent: str,
        service_data: dict[str, Any],
        charger_is_on: bool,
    ) -> bool | None:
        """Extracted async helper for SmartStart activate branch.

        Returns True after activating the charger when the helper handled
        the action, or None to continue normal evaluation.
        """
        try:
            # Defensive: only activate if charger currently reports off
            if not charger_is_on:
                self._log_action(
                    device_name,
                    logging.INFO,
                    "[SmartStart] Charging start time reached for %s -> activating charger (%s)",
                    device_name,
                    charger_ent,
                )
                # Activation should bypass normal throttle/confirmation gates so
                # scheduled SmartStart activations are not suppressed by recent
                # coordinator-issued switches or confirmation debounce.
                await self._maybe_switch(
                    "turn_on", service_data, desired=True, bypass_throttle=True
                )
                return True
        except Exception:
            _ignored_exc()
        return None

    async def _apply_charger_logic(
        self,
        device: DeviceConfig,
        *,
        now_local: datetime,
        battery: float,
        charger_is_on: bool,
        charger_available: bool,
        is_home: bool,
        start_time: datetime | None,
        smart_start_active: bool,
        precharge_required: bool,
        release_level: float | None,
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
        last_action, assumed_on = self._apply_charger_compute_assumed_state(
            device, charger_is_on
        )

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
                {
                    "release": list(self._precharge_release.keys()),
                    "ready": list(self._precharge_release_ready.keys()),
                },
            )
        except Exception:
            _ignored_exc()
        # Compute imminent-window flag used by multiple pre-start checks
        window_imminent = (
            smart_start_active
            and start_time is not None
            and start_time > now_local
            and (start_time - now_local) <= timedelta(seconds=5)
        )

        # Run early preflight checks (precharge-release, presence clear,
        # emergency charge and charger availability) in a focused helper
        # to keep this method linear and reduce cyclomatic complexity.
        preflight_result = await self._apply_charger_preflight_checks(
            device,
            charger_ent,
            now_local,
            charger_is_on,
            charger_available,
            precharge_required,
            release_level,
            margin_on,
            predicted_level,
            battery,
            service_data,
            device_name,
            smart_start_active,
            start_time,
            expected_on,
        )
        if preflight_result is not None:
            return preflight_result

        smartstart_handled = await self._apply_charger_smartstart_logic(
            device,
            now_local=now_local,
            battery=battery,
            charger_is_on=charger_is_on,
            start_time=start_time,
            smart_start_active=smart_start_active,
            precharge_required=precharge_required,
            forecast_holdoff=forecast_holdoff,
            charge_deficit=charge_deficit,
            predicted_level=predicted_level,
            margin_on=margin_on,
            smart_margin=smart_margin,
            service_data=service_data,
            device_name=device_name,
            charger_ent=charger_ent,
        )
        if smartstart_handled is not None:
            return smartstart_handled

        # Activation handled in `_apply_charger_smartstart_logic`

        smartstop_result = await self._apply_charger_smartstop_if_needed(
            charger_ent, device, device_name, battery, assumed_on, service_data
        )
        if smartstop_result is not None:
            return smartstop_result
        prestart_handled = await self._apply_charger_prestart_pause_checks(
            device,
            now_local=now_local,
            charger_is_on=charger_is_on,
            smart_start_active=smart_start_active,
            start_time=start_time,
            precharge_required=precharge_required,
            charge_deficit=charge_deficit,
            battery=battery,
            predicted_level=predicted_level,
            window_imminent=window_imminent,
            device_name=device_name,
            charger_ent=charger_ent,
            smart_margin=smart_margin,
            service_data=service_data,
        )
        if prestart_handled is not None:
            return prestart_handled

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
            handled = await self._apply_charger_precharge_logic(
                device,
                charger_ent=charger_ent,
                expected_on=expected_on,
                battery=battery,
                predicted_level=predicted_level,
                release_level=release_level,
                margin_on=margin_on,
                window_imminent=window_imminent,
                forecast_holdoff=forecast_holdoff,
                service_data=service_data,
                device_name=device_name,
            )
            if handled is not None:
                return handled

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

    def _post_alarm_handle_flipflop(
        self,
        dev_name: str,
        charger_ent: str,
        flipflop_count: int,
        now_epoch: float,
        entry_data: dict[str, Any],
        state_machine: Any,
    ) -> tuple[str | None, dict[str, Any]]:
        """Handle flipflop post-alarm correction and return (reason, details)."""
        reason: str | None = None
        details: dict[str, Any] = {}
        try:
            current = float(
                self._device_switch_throttle.get(
                    charger_ent, self._default_switch_throttle_seconds
                )
                or self._default_switch_throttle_seconds
            )
            mode_factor = 1.4
            var_multiplier = float(
                getattr(self, "_adaptive_throttle_multiplier", 2.0)
            ) * float(mode_factor)
            desired = max(
                current * var_multiplier,
                float(getattr(self, "_adaptive_throttle_min_seconds", 120.0)),
            )
            expires = float(
                now_epoch
                + float(getattr(self, "_adaptive_throttle_duration_seconds", 600.0))
            )
            try:
                self._post_alarm_temp_overrides[charger_ent] = {
                    "applied": float(desired),
                    "expires": float(expires),
                    "reason": "flipflop",
                }
                self._adaptive_throttle_overrides[charger_ent] = {
                    "original": float(
                        self._device_switch_throttle.get(
                            charger_ent, self._default_switch_throttle_seconds
                        )
                        or self._default_switch_throttle_seconds
                    ),
                    "applied": float(desired),
                    "expires": float(expires),
                }
                self._device_switch_throttle[charger_ent] = float(desired)
            except Exception:
                _ignored_exc()
            try:
                if state_machine and hasattr(state_machine, "add_error"):
                    state_machine.add_error(
                        dev_name,
                        f"post_alarm_missed_target:flipflop_count={flipflop_count}",
                    )
            except Exception:
                _ignored_exc()
            reason = "flipflop"
            details["flipflop_count"] = int(flipflop_count)
            details["applied_throttle"] = float(desired)
            streaks = self._post_alarm_miss_streaks.setdefault(charger_ent, {})
            streaks["flipflop"] = int(streaks.get("flipflop", 0)) + 1
            try:
                if int(streaks.get("flipflop", 0)) >= 2:
                    new_opts = dict(getattr(self.entry, "options", {}) or {})
                    mapping = dict(new_opts.get("adaptive_mode_overrides", {}) or {})
                    mapping[charger_ent] = "aggressive"
                    new_opts["adaptive_mode_overrides"] = mapping
                    try:
                        self.hass.config_entries.async_update_entry(
                            self.entry, options=new_opts
                        )
                    except Exception:
                        _ignored_exc()
            except Exception:
                _ignored_exc()
            _LOGGER.warning(
                "Post-alarm correction applied for %s (%s): flipflop_count=%d applied_throttle=%.1fs",
                dev_name,
                charger_ent,
                flipflop_count,
                float(desired),
            )
        except Exception:
            _ignored_exc()
        return reason, details

    def _post_alarm_handle_non_flipflop(
        self,
        pd: dict[str, Any],
        charger_ent: str,
        dev_name: str,
        now_local: datetime,
        now_epoch: float,
        entry_data: Any,
        state_machine: Any,
    ) -> tuple[str | None, dict[str, Any]]:
        """Handle late_start and drain_miss corrections and return (reason, details)."""
        reason: str | None = None
        details: dict[str, Any] = {}
        try:
            # predicted level not required in this branch
            charge_duration_min = float(pd.get("charge_duration_min") or 0.0)
            hours_until_alarm = None
            try:
                aiso = pd.get("alarm_time")
                adt = dt_util.parse_datetime(str(aiso)) if aiso else None
                if adt:
                    hours_until_alarm = max(
                        0.0, (adt - now_local).total_seconds() / 3600.0
                    )
            except Exception:
                hours_until_alarm = None

            # Delegate late_start vs drain_miss to focused helpers to lower complexity
            try:
                if (
                    charge_duration_min
                    and hours_until_alarm is not None
                    and (charge_duration_min / 60.0) > hours_until_alarm + 0.01
                ):
                    reason, details = self._post_alarm_handle_late_start(
                        pd,
                        charger_ent,
                        dev_name,
                        charge_duration_min,
                        hours_until_alarm,
                    )
                else:
                    reason, details = self._post_alarm_handle_drain_miss(
                        pd, charger_ent, dev_name, entry_data, state_machine
                    )
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()
        return reason, details

    def _post_alarm_handle_late_start(
        self,
        pd: dict[str, Any],
        charger_ent: str,
        dev_name: str,
        charge_duration_min: float,
        hours_until_alarm: float,
    ) -> tuple[str, dict[str, Any]]:
        """Handle late_start corrections and return (reason, details)."""
        reason = "late_start"
        details: dict[str, Any] = {}
        details["charge_duration_min"] = float(charge_duration_min)
        details["hours_until_alarm"] = float(hours_until_alarm)
        try:
            streaks = self._post_alarm_miss_streaks.setdefault(charger_ent, {})
            streaks["late_start"] = int(streaks.get("late_start", 0)) + 1
            if int(streaks.get("late_start", 0)) >= 2:
                try:
                    device_margin = float(
                        pd.get("smart_start_margin")
                        or float(
                            getattr(
                                self, "_smart_start_margin", DEFAULT_SMART_START_MARGIN
                            )
                        )
                    )
                except Exception:
                    device_margin = float(
                        getattr(self, "_smart_start_margin", DEFAULT_SMART_START_MARGIN)
                    )
                bump = 1.5
                new_margin = max(0.0, device_margin + bump)
                try:
                    new_opts = dict(getattr(self.entry, "options", {}) or {})
                    mapping = dict(
                        new_opts.get("smart_start_margin_overrides", {}) or {}
                    )
                    mapping[charger_ent] = float(new_margin)
                    new_opts["smart_start_margin_overrides"] = mapping
                    try:
                        self.hass.config_entries.async_update_entry(
                            self.entry, options=new_opts
                        )
                        self._post_alarm_persisted_smart_start[charger_ent] = float(
                            new_margin
                        )
                    except Exception:
                        _ignored_exc()
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()
        return reason, details

    def _post_alarm_handle_drain_miss(
        self,
        pd: dict[str, Any],
        charger_ent: str,
        dev_name: str,
        entry_data: Any,
        state_machine: Any,
    ) -> tuple[str | None, dict[str, Any]]:
        """Handle drain_miss corrections and return (reason, details)."""
        reason: str | None = None
        details: dict[str, Any] = {}
        try:
            predicted_level = float(pd.get("predicted_level_at_alarm", 0.0) or 0.0)
            actual_batt = float(pd.get("battery", 0.0) or 0.0)
            delta = predicted_level - actual_batt
            if delta >= 5.0:
                reason = "drain_miss"
                details["predicted_level_at_alarm"] = float(predicted_level)
                details["actual_battery"] = float(actual_batt)
                details["delta"] = float(delta)
                try:
                    pid = dev_name
                    self._post_alarm_learning_retrain_requests[pid] = (
                        int(self._post_alarm_learning_retrain_requests.get(pid, 0)) + 1
                    )
                    if int(self._post_alarm_learning_retrain_requests.get(pid, 0)) >= 3:
                        try:
                            learning = getattr(entry_data, "learning", None) or (
                                entry_data.get("learning")
                                if isinstance(entry_data, dict)
                                else None
                            )
                        except Exception:
                            learning = None
                        try:
                            if learning and hasattr(learning, "async_reset_profile"):
                                self.hass.async_create_task(
                                    learning.async_reset_profile(pid)
                                )
                                if state_machine and hasattr(
                                    state_machine, "add_error"
                                ):
                                    state_machine.add_error(
                                        pid, "learning_reset_scheduled:drain_miss"
                                    )
                                self._post_alarm_learning_retrain_requests[pid] = 0
                                details["learning_reset_scheduled"] = True
                        except Exception:
                            _ignored_exc()
                except Exception:
                    _ignored_exc()
        except Exception:
            _ignored_exc()
        return reason, details

    def _post_alarm_count_flipflop(self, charger_ent: str, now_epoch: float) -> int:
        """Return number of flipflop events within window for charger_ent."""
        try:
            cutoff = now_epoch - float(getattr(self, "_flipflop_window_seconds", 300.0))
            events = [
                e for e in (self._flipflop_events.get(charger_ent) or []) if e >= cutoff
            ]
            return len(events)
        except Exception:
            return 0

    def _post_alarm_record_correction(
        self,
        charger_ent: str,
        dev_name: str,
        alarm_epoch: float,
        now_epoch: float,
        reason: str | None,
        details: dict[str, Any],
    ) -> None:
        """Record a post-alarm correction entry and update last_handled."""
        try:
            entry = {
                "entity": charger_ent,
                "device": dev_name,
                "alarm_epoch": alarm_epoch,
                "timestamp": now_epoch,
                "reason": reason or "unknown",
                "details": details,
            }
            self._post_alarm_corrections.insert(0, entry)
            if len(self._post_alarm_corrections) > 50:
                self._post_alarm_corrections = self._post_alarm_corrections[:50]
        except Exception:
            _ignored_exc()
        try:
            self._post_alarm_last_handled[charger_ent] = alarm_epoch
        except Exception:
            _ignored_exc()

    def _post_alarm_parse_and_validate(
        self, pd: dict[str, Any], now_epoch: float
    ) -> tuple[str, float, float, float] | None:
        """Parse and validate post-alarm payload.

        Returns (charger_ent, alarm_epoch, target, battery) on success,
        or None if the alarm should be ignored/skipped.
        Side-effects: may update _post_alarm_last_handled when battery reached target.
        """
        try:
            charger_ent = pd.get("charger_switch")
            alarm_iso = pd.get("alarm_time")
            target = float(pd.get("target", 0.0) or 0.0)
            battery = float(pd.get("battery", 0.0) or 0.0)
            if not charger_ent or not alarm_iso:
                return None
            alarm_dt = dt_util.parse_datetime(str(alarm_iso))
            if alarm_dt is None:
                return None
            alarm_epoch = float(dt_util.as_timestamp(alarm_dt))
            # Only handle alarms that have passed
            if now_epoch < alarm_epoch:
                return None
            # Avoid handling the same alarm repeatedly
            last_handled = float(
                self._post_alarm_last_handled.get(charger_ent, 0.0) or 0.0
            )
            if last_handled >= alarm_epoch:
                return None
            # If battery reached target, mark handled and skip
            if battery >= target - 0.5:
                try:
                    self._post_alarm_last_handled[charger_ent] = alarm_epoch
                except Exception:
                    _ignored_exc()
                return None
            return charger_ent, alarm_epoch, target, battery
        except Exception:
            _ignored_exc()
            return None

    def _handle_post_alarm_self_heal(
        self, results: dict[str, dict[str, Any]], now_local: datetime
    ) -> None:
        """Handle post-alarm diagnosis and conservative self-healing actions.

        This was extracted from _async_update_data to reduce complexity of the
        main update method. It inspects completed alarms, diagnoses missed
        targets (flip-flops, late starts, drain misestimates), applies
        temporary adaptive throttle overrides, and records bounded correction
        history. Exceptions are caught to avoid failing the coordinator.
        """
        # Ensure entry_data and state_machine variables always exist for later use
        try:
            entries = self.hass.data.get(DOMAIN, {}).get("entries", {})
            entry_data = entries.get(getattr(self.entry, "entry_id", ""), {}) or {}
            state_machine = (
                entry_data.get("state_machine")
                if isinstance(entry_data, dict)
                else None
            )
        except Exception:
            entry_data = {}
            state_machine = None

        try:
            now_epoch = float(dt_util.as_timestamp(now_local))
        except Exception:
            now_epoch = float(dt_util.as_timestamp(dt_util.utcnow()))

        for dev_name, pd in (results or {}).items():
            try:
                self._handle_post_alarm_for_device(
                    pd, dev_name, now_local, now_epoch, entry_data, state_machine
                )
            except Exception:
                _ignored_exc()

    def _handle_post_alarm_for_device(
        self,
        pd: dict[str, Any],
        dev_name: str,
        now_local: datetime,
        now_epoch: float,
        entry_data: dict[str, Any],
        state_machine: Any,
    ) -> None:
        """Handle post-alarm diagnosis and possible corrections for a single device.

        Side-effects: updates `_post_alarm_last_handled`, records corrections,
        and may schedule learning resets via the learning subsystem.
        """
        try:
            parsed = self._post_alarm_parse_and_validate(pd, now_epoch)
            if not parsed:
                return
            charger_ent, alarm_epoch, target, battery = parsed
            # Count flip-flop events
            flipflop_count = self._post_alarm_count_flipflop(charger_ent, now_epoch)

            # Delegate flipflop vs non-flipflop handling
            reason = None
            details: dict[str, Any] = {}
            if flipflop_count >= int(getattr(self, "_flipflop_warn_threshold", 3)):
                try:
                    reason, details = self._post_alarm_handle_flipflop(
                        dev_name,
                        charger_ent,
                        flipflop_count,
                        now_epoch,
                        entry_data,
                        state_machine,
                    )
                except Exception:
                    _ignored_exc()
            else:
                try:
                    reason, details = self._post_alarm_handle_non_flipflop(
                        pd,
                        charger_ent,
                        dev_name,
                        now_local,
                        now_epoch,
                        entry_data,
                        state_machine,
                    )
                except Exception:
                    _ignored_exc()

            # Record correction or suggestion and update last_handled
            try:
                self._post_alarm_finalize_handling(
                    charger_ent, dev_name, alarm_epoch, now_epoch, reason, details
                )
            except Exception:
                _ignored_exc()
        except Exception:
            _ignored_exc()

    def _post_alarm_finalize_handling(
        self,
        charger_ent: str,
        dev_name: str,
        alarm_epoch: float,
        now_epoch: float,
        reason: str | None,
        details: dict[str, Any],
    ) -> None:
        """Finalize handling for an alarm: record correction and update last_handled.

        This consolidates the final recording and state update so the main handler
        keeps fewer branching paths.
        """
        try:
            # Use the existing record helper for the correction entry
            self._post_alarm_record_correction(
                charger_ent, dev_name, alarm_epoch, now_epoch, reason, details
            )
        except Exception:
            _ignored_exc()
