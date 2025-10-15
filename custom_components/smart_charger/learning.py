from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    LEARNING_CACHE_TTL,
    LEARNING_DEFAULT_SPEED,
    LEARNING_EMA_ALPHA,
    LEARNING_MAX_SPEED,
    LEARNING_MIN_SPEED,
    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    UNKNOWN_STATES,
)

_LOGGER = logging.getLogger(__name__)


"""Half-life (hours) used when decaying older measurements."""
DECAY_HALF_LIFE_HOURS = 12
SAVE_DEBOUNCE_SECONDS = 10
SESSION_RETRY_DELAYS: tuple[int, ...] = (30, 90, 300)
MIN_SESSION_DELTA = 0.2
PROFILE_SCHEMA_VERSION = 3
STORAGE_KEY_LEGACY = f"{DOMAIN}_learning"
MAX_SAMPLES_DEFAULT = 200
MAX_CYCLES_DEFAULT = 120


def _time_bucket(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _ema_update(
    previous: Optional[float], value: float, alpha: float = LEARNING_EMA_ALPHA
) -> float:
    if previous is None:
        return value
    return alpha * value + (1 - alpha) * previous


def _decay_to_baseline(
    value: float, hours_old: Optional[float], baseline: float = 1.0
) -> float:
    if hours_old is None or hours_old <= 0:
        return value
    factor = 0.5 ** (hours_old / DECAY_HALF_LIFE_HOURS)
    return baseline + (value - baseline) * factor


class SmartChargerLearning:
    """Persistent learning engine tracking charge speeds and cycles."""

    STORAGE_VERSION = 1

    def __init__(self, hass, entry_id: Optional[str] = None) -> None:
        self.hass = hass
        self._entry_id = entry_id
        storage_key = (
            f"{STORAGE_KEY_LEGACY}_{entry_id}" if entry_id else STORAGE_KEY_LEGACY
        )
        self._store = Store(hass, self.STORAGE_VERSION, storage_key)
        self._legacy_store = (
            None
            if not entry_id
            else Store(hass, self.STORAGE_VERSION, STORAGE_KEY_LEGACY)
        )
        self._data: Dict[str, Any] = self._default_storage()
        self._save_debounce_unsub: Optional[Callable[[], None]] = None
        self._session_retry_unsubs: Dict[str, Callable[[], None]] = {}
        self._avg_cache: Dict[str, Tuple[float, float]] = {}
        self._lock = asyncio.Lock()
        self._migrated_from_legacy = False
        self._recent_sample_max_age_hours = DEFAULT_LEARNING_RECENT_SAMPLE_HOURS

    def _default_meta(self) -> Dict[str, Any]:
        now_iso = dt_util.utcnow().isoformat()
        return {
            "model_revision": PROFILE_SCHEMA_VERSION,
            "entry_id": self._entry_id,
            "created": now_iso,
            "updated": now_iso,
            "profile_count": 0,
            "sample_count": 0,
            "cycle_count": 0,
        }

    def _default_storage(self) -> Dict[str, Any]:
        return {"meta": self._default_meta(), "profiles": {}}

    @property
    def _profiles(self) -> Dict[str, Dict[str, Any]]:
        return self._data.setdefault("profiles", {})

    @property
    def _meta(self) -> Dict[str, Any]:
        return self._data.setdefault("meta", self._default_meta())

    def _refresh_meta(self) -> None:
        profiles = self._profiles
        meta = self._meta
        meta["profile_count"] = len(profiles)
        meta["sample_count"] = sum(len(p.get("samples", [])) for p in profiles.values())
        meta["cycle_count"] = sum(len(p.get("cycles", [])) for p in profiles.values())
        meta["updated"] = dt_util.utcnow().isoformat()

    async def async_load(self, profile_id: Optional[str] = None) -> None:
        try:
            data = await self._store.async_load()
            if not data and self._legacy_store is not None:
                legacy = await self._legacy_store.async_load()
                if legacy:
                    data = legacy
                    self._migrated_from_legacy = True

            if not data:
                self._data = self._default_storage()
                return

            if "profiles" not in data or not isinstance(data.get("profiles"), dict):
                profiles = data if isinstance(data, dict) else {}
                migrated = self._default_storage()
                migrated["profiles"].update(profiles)
                data = migrated

            self._data = data
            if self._meta.get("model_revision", 0) < PROFILE_SCHEMA_VERSION:
                self._meta["model_revision"] = PROFILE_SCHEMA_VERSION

            if profile_id and profile_id in self._profiles:
                self._ensure_profile_schema(profile_id)
            else:
                for pid in list(self._profiles.keys()):
                    self._ensure_profile_schema(pid)

            self._refresh_meta()
            self._avg_cache.clear()
            _LOGGER.debug(
                "Learning data loaded (profiles=%d, migrated=%s)",
                len(self._profiles),
                self._migrated_from_legacy,
            )

            if self._migrated_from_legacy:
                await self.async_save()
        except Exception:
            _LOGGER.exception("Learning: load failed")

    async def async_save(self) -> None:
        try:
            if self._save_debounce_unsub:
                self._save_debounce_unsub()
                self._save_debounce_unsub = None
            async with self._lock:
                self._refresh_meta()
                await self._store.async_save(self._data)
        except Exception:
            _LOGGER.exception("Learning: save failed")

    async def ensure_profile(self, profile_id: str) -> None:
        """Ensure profile exists in memory."""
        created = False
        async with self._lock:
            if profile_id not in self._profiles:
                self._profiles[profile_id] = self._default_profile()
                created = True
            else:
                self._ensure_profile_schema(profile_id)
            if created:
                self._refresh_meta()

        if created:
            await self.async_save()

        self._invalidate_cache(profile_id)

    def _avg_cache_key(self, profile_id: Optional[str], scope: str) -> str:
        return f"{profile_id or '__global__'}::{scope}"

    def _avg_cache_get(self, profile_id: Optional[str], scope: str) -> Optional[float]:
        token = self._avg_cache.get(self._avg_cache_key(profile_id, scope))
        if not token:
            return None
        value, timestamp = token
        if (dt_util.utcnow().timestamp() - timestamp) > LEARNING_CACHE_TTL:
            self._avg_cache.pop(self._avg_cache_key(profile_id, scope), None)
            return None
        return value

    def _avg_cache_set(
        self, profile_id: Optional[str], scope: str, value: float
    ) -> None:
        self._avg_cache[self._avg_cache_key(profile_id, scope)] = (
            value,
            dt_util.utcnow().timestamp(),
        )

    def _entry_speed_value(
        self, entry: Optional[Dict[str, Any]], now: datetime
    ) -> Optional[float]:
        if not entry:
            return None
        ema = entry.get("ema")
        if ema is None:
            return None
        age_hours: Optional[float] = None
        last = entry.get("last_sample")
        if last:
            parsed = dt_util.parse_datetime(last)
            if parsed:
                age_hours = (now - parsed).total_seconds() / 3600.0
        value = _decay_to_baseline(float(ema), age_hours)
        return round(self._clamp_speed(value), 3)

    def _profile_bucket_avg(
        self, profile_id: str, bucket_key: str, now: datetime
    ) -> Optional[float]:
        pdata = self._ensure_profile_schema(profile_id)
        bucket_entry = pdata.get("bucket_stats", {}).get(bucket_key)
        return self._entry_speed_value(bucket_entry, now)

    def _profile_overall_avg(self, profile_id: str, now: datetime) -> Optional[float]:
        pdata = self._ensure_profile_schema(profile_id)
        return self._entry_speed_value(pdata.get("stats"), now)

    def _global_profile_avg(self, now: datetime) -> Optional[float]:
        values: list[float] = []
        for pid in list(self._profiles.keys()):
            value = self._profile_overall_avg(pid, now)
            if value is not None:
                values.append(value)
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    def _collect_samples(self, profile_id: Optional[str]) -> list[tuple[str, float]]:
        if profile_id and profile_id in self._profiles:
            return list(self._profiles[profile_id].get("samples", []))
        samples: list[tuple[str, float]] = []
        for pdata in self._profiles.values():
            samples.extend(pdata.get("samples", []))
        return samples

    def _latest_sample_speed(
        self, profile_id: Optional[str], now: datetime
    ) -> Optional[float]:
        samples = self._collect_samples(profile_id)
        if not samples:
            return None
        ts, speed = samples[-1]
        parsed = dt_util.parse_datetime(ts)
        if not parsed:
            return None
        age_hours = (now - parsed).total_seconds() / 3600.0
        if age_hours <= self._recent_sample_max_age_hours:
            return round(self._clamp_speed(float(speed)), 3)
        return None

    def _recent_sample_average(
        self, profile_id: Optional[str], now: datetime
    ) -> Optional[float]:
        samples = self._collect_samples(profile_id)
        if not samples:
            return None
        weighted_sum = 0.0
        total_weight = 0.0
        for ts, speed in samples[-100:]:
            parsed = dt_util.parse_datetime(ts)
            if not parsed:
                continue
            hours_old = (now - parsed).total_seconds() / 3600.0
            weight = 0.5 ** (hours_old / (DECAY_HALF_LIFE_HOURS / 2))
            weighted_sum += float(speed) * weight
            total_weight += weight
        if total_weight <= 0:
            return None
        avg = weighted_sum / total_weight
        return round(self._clamp_speed(avg), 3)

    def avg_speed(self, profile_id: Optional[str] = None) -> float:
        """Compute a weighted average speed that favors recent, time-matched samples."""
        try:
            now = dt_util.now()
            bucket_key = _time_bucket(now.hour)
            if profile_id and profile_id in self._profiles:
                fresh_sample = self._latest_sample_speed(profile_id, now)
                if fresh_sample is not None:
                    self._invalidate_cache(profile_id)
                    self._avg_cache_set(profile_id, bucket_key, fresh_sample)
                    self._avg_cache_set(profile_id, "profile", fresh_sample)
                    return fresh_sample
                cached_bucket = self._avg_cache_get(profile_id, bucket_key)
                if cached_bucket is not None:
                    return cached_bucket
                bucket_avg = self._profile_bucket_avg(profile_id, bucket_key, now)
                if bucket_avg is not None:
                    self._avg_cache_set(profile_id, bucket_key, bucket_avg)
                    return bucket_avg
                profile_avg = self._profile_overall_avg(profile_id, now)
                if profile_avg is not None:
                    self._avg_cache_set(profile_id, "profile", profile_avg)
                    return profile_avg

            cached_global = self._avg_cache_get(None, "global")
            if cached_global is not None:
                return cached_global

            global_avg = self._global_profile_avg(now)
            if global_avg is not None:
                self._avg_cache_set(None, "global", global_avg)
                return global_avg

            fallback = self._recent_sample_average(profile_id, now)
            if fallback is not None:
                self._avg_cache_set(profile_id, "fallback", fallback)
                return fallback

            return LEARNING_DEFAULT_SPEED
        except Exception:
            _LOGGER.exception("Adaptive avg_speed calculation failed")
            return LEARNING_DEFAULT_SPEED

    def set_recent_sample_window(self, hours: float) -> None:
        """Update the time window considered "recent" for charge samples."""
        try:
            value = float(hours)
        except (TypeError, ValueError):
            return
        value = max(0.25, min(48.0, value))
        if abs(value - self._recent_sample_max_age_hours) < 1e-6:
            return
        self._recent_sample_max_age_hours = value
        self._avg_cache.clear()

    async def async_start_session(
        self, profile_id: str, level_now: float, sensor: Optional[str] = None
    ) -> None:
        """Start tracking a new charging session."""

        async with self._lock:
            pdata = self._ensure_profile_schema(profile_id)
            pdata["current_session"] = {
                "started": dt_util.now().isoformat(),
                "level": float(level_now),
                "sensor": sensor,
                "retries": 0,
            }
            self._refresh_meta()

        self._cancel_session_retry(profile_id)
        self._invalidate_cache(profile_id)
        self._schedule_save()
        _LOGGER.debug(
            "Session started for %s at %.1f%% (sensor=%s)",
            profile_id,
            level_now,
            sensor,
        )

    def start_session(
        self, profile_id: str, level_now: float, sensor: Optional[str] = None
    ) -> None:
        """Compat wrapper scheduling the async session start."""
        self.hass.async_create_task(
            self.async_start_session(profile_id, level_now, sensor)
        )

    async def end_session(
        self,
        profile_id: str,
        level_end: Optional[float] = None,
        sensor: Optional[str] = None,
    ) -> None:
        """Finish tracking the active charging session."""
        profile = self._ensure_profile_schema(profile_id)
        session = self._normalize_session(profile.get("current_session"))
        if not session:
            _LOGGER.debug("No active session for %s", profile_id)
            return

        profile["current_session"] = session
        if sensor:
            session["sensor"] = sensor

        sensor_id = session.get("sensor")
        level_now = self._read_battery_sensor(sensor_id)
        if level_now is None and level_end is not None:
            try:
                level_now = float(level_end)
            except (TypeError, ValueError):
                level_now = None

        if level_now is None:
            _LOGGER.debug(
                "Deferring session finalization for %s: sensor=%s unavailable",
                profile_id,
                sensor_id,
            )
            self._handle_session_retry(profile_id, profile, session)
            return

        try:
            start_iso = session.get("started")
            parsed = (
                dt_util.parse_datetime(start_iso)
                if isinstance(start_iso, str)
                else None
            )
            start_time = parsed or dt_util.now()
            start_level = float(session.get("level", level_now))
        except Exception:
            _LOGGER.exception("Error while preparing session data for %s", profile_id)
            profile.pop("current_session", None)
            self._cancel_session_retry(profile_id)
            return

        success = await self.record_cycle(
            profile_id=profile_id,
            start_time=start_time,
            end_time=dt_util.now(),
            start_level=start_level,
            end_level=float(level_now),
            reached_target=start_level < level_now,
            error=None,
        )

        if success:
            async with self._lock:
                profile = self._ensure_profile_schema(profile_id)
                profile.pop("current_session", None)
                self._refresh_meta()
            self._cancel_session_retry(profile_id)
        else:
            self._handle_session_retry(profile_id, profile, session)

    async def record_cycle(
        self,
        profile_id: str,
        start_time: datetime,
        end_time: datetime,
        start_level: float,
        end_level: float,
        reached_target: bool,
        error: Optional[str],
    ) -> bool:
        """Persist a completed charging cycle and update derived metrics."""
        duration_min = max(0.0, (end_time - start_time).total_seconds() / 60.0)
        duration_hours = duration_min / 60.0
        try:
            delta = max(0.0, end_level - start_level)
            speed = delta / duration_hours if duration_hours > 0 else 0.0
        except Exception:
            delta = 0.0
            speed = 0.0

        if delta < MIN_SESSION_DELTA or speed <= 0 or speed > 500:
            _LOGGER.debug(
                "Ignoring implausible session for %s: Δ%.3f%% over %.1fmin (speed=%.3f%%/h)",
                profile_id,
                delta,
                duration_min,
                speed,
            )
            return False

        accepted = False
        async with self._lock:
            profile = self._ensure_profile_schema(profile_id)
            stats = profile.get("stats", {})
            baseline = stats.get("ema")
            if baseline and speed > self._clamp_speed(float(baseline) * 3):
                _LOGGER.debug(
                    "Dropping outlier session for %s: speed %.3f too far from baseline %.3f",
                    profile_id,
                    speed,
                    baseline,
                )
                return False

            speed = self._clamp_speed(speed)
            timestamp = dt_util.now().isoformat()
            profile.setdefault("samples", []).append((timestamp, speed))
            profile.setdefault("cycles", []).append(
                {
                    "start_time": getattr(
                        start_time, "isoformat", lambda: start_time
                    )(),
                    "end_time": getattr(end_time, "isoformat", lambda: end_time)(),
                    "start_level": round(start_level, 1),
                    "end_level": round(end_level, 1),
                    "duration_min": round(duration_min, 1),
                    "speed": round(speed, 3),
                    "reached_target": reached_target,
                    "error_cause": error,
                }
            )
            self._trim_profile(profile)
            self._update_stats(profile, speed, timestamp, start_time)
            self._refresh_meta()
            accepted = True

        if accepted:
            _LOGGER.debug(
                "Recorded cycle for %s: Δ%.1f%% in %.1fmin (%.3f %%/h)",
                profile_id,
                end_level - start_level,
                duration_min,
                speed,
            )
            self._schedule_save()
            self._invalidate_cache(profile_id)
        return accepted

    def _default_profile(self) -> Dict[str, Any]:
        return {
            "version": PROFILE_SCHEMA_VERSION,
            "samples": [],
            "cycles": [],
            "stats": {"ema": None, "count": 0, "last_sample": None},
            "bucket_stats": {},
        }

    def _ensure_profile_schema(self, profile_id: str) -> Dict[str, Any]:
        profiles = self._profiles
        pdata = profiles.get(profile_id)
        if not isinstance(pdata, dict):
            pdata = self._default_profile()
            profiles[profile_id] = pdata

        pdata.setdefault("version", PROFILE_SCHEMA_VERSION)
        current_version = int(pdata.get("version", 0))
        if current_version < 3:
            self._migrate_profile_to_hourly_speeds(pdata)
            current_version = 3
        if current_version < PROFILE_SCHEMA_VERSION:
            current_version = PROFILE_SCHEMA_VERSION
        pdata["version"] = current_version

        pdata.setdefault("samples", [])
        pdata.setdefault("cycles", [])
        pdata.setdefault("stats", {"ema": None, "count": 0, "last_sample": None})
        pdata.setdefault("bucket_stats", {})
        if "current_session" in pdata:
            normalized = self._normalize_session(pdata.get("current_session"))
            if normalized:
                pdata["current_session"] = normalized
            else:
                pdata.pop("current_session", None)
        return pdata

    def _migrate_profile_to_hourly_speeds(self, profile: Dict[str, Any]) -> None:
        def _convert_speed(value: Any) -> Optional[float]:
            try:
                return self._clamp_speed(float(value) * 60.0)
            except (TypeError, ValueError):
                return None

        samples = profile.get("samples")
        if isinstance(samples, list):
            converted_samples: list[Any] = []
            for entry in samples:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    converted = _convert_speed(entry[1])
                    if converted is not None:
                        converted_samples.append((entry[0], converted))
                        continue
                converted_samples.append(entry)
            profile["samples"] = converted_samples

        cycles = profile.get("cycles")
        if isinstance(cycles, list):
            for entry in cycles:
                if isinstance(entry, dict) and "speed" in entry:
                    converted = _convert_speed(entry.get("speed"))
                    if converted is not None:
                        entry["speed"] = round(converted, 3)

        stats = profile.get("stats")
        if isinstance(stats, dict):
            converted = _convert_speed(stats.get("ema"))
            if converted is not None:
                stats["ema"] = converted

        bucket_stats = profile.get("bucket_stats")
        if isinstance(bucket_stats, dict):
            for bucket in list(bucket_stats.keys()):
                entry = bucket_stats.get(bucket)
                if isinstance(entry, dict):
                    converted = _convert_speed(entry.get("ema"))
                    if converted is not None:
                        entry["ema"] = converted

    def _invalidate_cache(self, profile_id: Optional[str]) -> None:
        if not self._avg_cache:
            return
        if profile_id is None:
            keys_to_drop = list(self._avg_cache.keys())
        else:
            prefix = f"{profile_id}::"
            keys_to_drop = [
                key
                for key in self._avg_cache
                if key.startswith(prefix) or key.startswith("__global__::")
            ]
        for key in keys_to_drop:
            self._avg_cache.pop(key, None)

    @staticmethod
    def _clamp_speed(value: float) -> float:
        return max(LEARNING_MIN_SPEED, min(LEARNING_MAX_SPEED, value))

    def _normalize_session(self, session: Any) -> Optional[Dict[str, Any]]:
        if not session:
            return None
        if isinstance(session, dict):
            session.setdefault("retries", 0)
            session.setdefault("sensor", None)
            return session
        if isinstance(session, (list, tuple)) and len(session) >= 2:
            try:
                level = float(session[1])
            except (TypeError, ValueError):
                level = 0.0
            return {
                "started": session[0],
                "level": level,
                "sensor": None,
                "retries": 0,
            }
        return None

    def _read_battery_sensor(self, sensor: Optional[str]) -> Optional[float]:
        if not sensor:
            return None
        state = self.hass.states.get(sensor)
        if not state or state.state in UNKNOWN_STATES:
            return None
        try:
            return float(state.state)
        except (TypeError, ValueError):
            _LOGGER.debug("Cannot parse battery state %s for %s", state.state, sensor)
            return None

    def _update_stats(
        self,
        profile: Dict[str, Any],
        speed: float,
        timestamp: str,
        start_time: datetime,
    ) -> None:
        stats = profile.setdefault(
            "stats", {"ema": None, "count": 0, "last_sample": None}
        )
        stats["ema"] = _ema_update(stats.get("ema"), speed)
        stats["count"] = int(stats.get("count", 0)) + 1
        stats["last_sample"] = timestamp

        bucket_map = profile.setdefault("bucket_stats", {})
        local_start = dt_util.as_local(start_time)
        bucket_key = _time_bucket(local_start.hour)
        bucket_entry = bucket_map.setdefault(
            bucket_key, {"ema": None, "count": 0, "last_sample": None}
        )
        bucket_entry["ema"] = _ema_update(
            bucket_entry.get("ema"), speed, alpha=LEARNING_EMA_ALPHA
        )
        bucket_entry["count"] = int(bucket_entry.get("count", 0)) + 1
        bucket_entry["last_sample"] = timestamp

    def _trim_profile(
        self,
        profile: Dict[str, Any],
        max_samples: int = MAX_SAMPLES_DEFAULT,
        max_cycles: int = MAX_CYCLES_DEFAULT,
    ) -> None:
        if max_samples > 0 and "samples" in profile:
            profile["samples"] = profile["samples"][-max_samples:]
        if max_cycles > 0 and "cycles" in profile:
            profile["cycles"] = profile["cycles"][-max_cycles:]
        bucket_stats = profile.get("bucket_stats")
        if isinstance(bucket_stats, dict):
            for bucket, entry in list(bucket_stats.items()):
                if not entry.get("ema"):
                    bucket_stats.pop(bucket, None)

    async def async_cleanup_old_data(
        self,
        max_samples: int = MAX_SAMPLES_DEFAULT,
        max_cycles: int = MAX_CYCLES_DEFAULT,
    ) -> None:
        """Keep only a bounded number of stored samples and cycles."""
        async with self._lock:
            for pdata in self._profiles.values():
                self._trim_profile(pdata, max_samples, max_cycles)
            self._refresh_meta()

        _LOGGER.debug(
            "Cleaned up learning data (max %d samples, %d cycles)",
            max_samples,
            max_cycles,
        )
        self._schedule_save()
        self._avg_cache.clear()

    def cleanup_old_data(
        self,
        max_samples: int = MAX_SAMPLES_DEFAULT,
        max_cycles: int = MAX_CYCLES_DEFAULT,
    ) -> None:
        """Compat wrapper that schedules cleanup asynchronously."""
        self.hass.async_create_task(
            self.async_cleanup_old_data(max_samples=max_samples, max_cycles=max_cycles)
        )

    def snapshot(self, profile_id: Optional[str] = None) -> Dict[str, Any]:
        """Return a deep copy of the current learning state for diagnostics."""
        meta_copy = copy.deepcopy(self._meta)
        meta_copy["snapshot_ts"] = dt_util.utcnow().isoformat()
        if profile_id:
            profiles = {}
            pdata = self._profiles.get(profile_id)
            if pdata:
                profiles[profile_id] = copy.deepcopy(pdata)
        else:
            profiles = copy.deepcopy(self._profiles)
        return {"meta": meta_copy, "profiles": profiles}

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self._profiles.keys())

    async def async_reset_profile(self, profile_id: str) -> None:
        """Reset a single profile to its default state."""
        async with self._lock:
            if profile_id in self._profiles:
                self._profiles[profile_id] = self._default_profile()
                self._refresh_meta()
        await self.async_save()
        self._invalidate_cache(profile_id)

    async def async_reset_all(self) -> None:
        """Clear all learning data."""
        async with self._lock:
            self._data = self._default_storage()
        await self.async_save()
        self._invalidate_cache(None)

    def _handle_session_retry(
        self,
        profile_id: str,
        profile: Dict[str, Any],
        session: Dict[str, Any],
    ) -> None:
        sensor = session.get("sensor")
        attempt = int(session.get("retries", 0))
        start_level_raw = session.get("level", 0.0)
        try:
            start_level = float(start_level_raw)
        except (TypeError, ValueError):
            start_level = 0.0
        if not sensor:
            _LOGGER.debug(
                "Cannot retry session for %s: battery sensor unknown", profile_id
            )
            profile.pop("current_session", None)
            self._cancel_session_retry(profile_id)
            return
        if attempt >= len(SESSION_RETRY_DELAYS):
            _LOGGER.debug(
                "Giving up on session for %s after %d attempts (sensor=%s, start_level=%.1f%%)",
                profile_id,
                attempt,
                sensor,
                start_level,
            )
            profile.pop("current_session", None)
            self._cancel_session_retry(profile_id)
            return

        delay = SESSION_RETRY_DELAYS[attempt]
        session["retries"] = attempt + 1
        profile["current_session"] = session
        self._schedule_session_retry(profile_id, delay)
        _LOGGER.debug(
            "Retrying session finalize for %s in %ss (attempt %d/%d, sensor=%s)",
            profile_id,
            delay,
            attempt + 1,
            len(SESSION_RETRY_DELAYS),
            sensor,
        )

    def _schedule_session_retry(self, profile_id: str, delay: float) -> None:
        self._cancel_session_retry(profile_id)

        def _callback(_now: Any) -> None:
            self._session_retry_unsubs.pop(profile_id, None)
            asyncio.run_coroutine_threadsafe(
                self._retry_end_session(profile_id), self.hass.loop
            )

        self._session_retry_unsubs[profile_id] = async_call_later(
            self.hass,
            delay,
            _callback,
        )

    def _cancel_session_retry(self, profile_id: str) -> None:
        unsub = self._session_retry_unsubs.pop(profile_id, None)
        if unsub:
            try:
                unsub()
            except Exception:
                _LOGGER.debug("Failed to cancel retry timer for %s", profile_id)

    async def _retry_end_session(self, profile_id: str) -> None:
        profile = self._ensure_profile_schema(profile_id)
        session = self._normalize_session(profile.get("current_session"))
        if not session:
            return
        profile["current_session"] = session
        sensor = session.get("sensor")
        level = self._read_battery_sensor(sensor)
        await self.end_session(profile_id, level_end=level, sensor=sensor)

    def _schedule_save(self) -> None:
        if self._save_debounce_unsub:
            self._save_debounce_unsub()

        def _async_save_callback(_now: Any) -> None:
            self._save_debounce_unsub = None
            asyncio.run_coroutine_threadsafe(self.async_save(), self.hass.loop)

        self._save_debounce_unsub = async_call_later(
            self.hass,
            SAVE_DEBOUNCE_SECONDS,
            _async_save_callback,
        )
