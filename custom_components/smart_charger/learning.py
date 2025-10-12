from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import logging
from datetime import datetime
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


EMA_ALPHA = 0.35  # Gewicht für neue Messungen
DECAY_HALF_LIFE_HOURS = 48  # Halbwertszeit für Verblassen alter Messungen
SAVE_DEBOUNCE_SECONDS = 10


def _time_bucket(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _ema_update(previous: Optional[float], value: float, alpha: float = EMA_ALPHA) -> float:
    if previous is None:
        return value
    return alpha * value + (1 - alpha) * previous


def _decay_to_baseline(value: float, hours_old: Optional[float], baseline: float = 1.0) -> float:
    if hours_old is None or hours_old <= 0:
        return value
    factor = 0.5 ** (hours_old / DECAY_HALF_LIFE_HOURS)
    return baseline + (value - baseline) * factor


class SmartChargerLearning:
    """Persistente Lern-Engine für Ladegeschwindigkeiten und Zyklen."""

    STORAGE_VERSION = 1
    STORAGE_KEY = f"{DOMAIN}_learning"

    def __init__(self, hass) -> None:
        self.hass = hass
        self._store = Store(hass, self.STORAGE_VERSION, self.STORAGE_KEY)
        # _data: {profile_id: {"samples": [(ts, speed)], "cycles": [...], "current_session": (ts, level)}}
        self._data: Dict[str, Dict[str, Any]] = {}
        self._save_debounce_unsub: Optional[Callable[[], None]] = None

    # -------------------------------------------------------------------------
    # Loading & Saving
    # -------------------------------------------------------------------------
    async def async_load(self, profile_id: Optional[str] = None) -> None:
        try:
            data = await self._store.async_load()
            if not data:
                return
            if profile_id and profile_id in data:
                self._data[profile_id] = data[profile_id]
                self._ensure_profile_schema(profile_id)
            else:
                self._data = data
                for pid in list(self._data.keys()):
                    self._ensure_profile_schema(pid)
            _LOGGER.debug("Learning data loaded (profiles=%d)", len(self._data))
        except Exception:
            _LOGGER.exception("Learning: load failed")

    async def async_save(self) -> None:
        try:
            if self._save_debounce_unsub:
                self._save_debounce_unsub()
                self._save_debounce_unsub = None
            await self._store.async_save(self._data)
        except Exception:
            _LOGGER.exception("Learning: save failed")

    async def ensure_profile(self, profile_id: str) -> None:
        """Ensure profile exists in memory."""
        if profile_id not in self._data:
            self._data[profile_id] = self._default_profile()
            await self.async_save()
        else:
            self._ensure_profile_schema(profile_id)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    def avg_speed(self, profile_id: Optional[str] = None) -> float:
        """Berechne gewichteten Durchschnitt in %/min, stärker gewichtet nach Aktualität und Tageszeit."""
        try:
            now = dt_util.now()

            def _from_entry(entry: Optional[Dict[str, Any]]) -> Optional[float]:
                if not entry:
                    return None
                ema = entry.get("ema")
                if ema is None:
                    return None
                last = entry.get("last_sample")
                age_hours: Optional[float] = None
                if last:
                    dt = dt_util.parse_datetime(last)
                    if dt:
                        age_hours = (now - dt).total_seconds() / 3600.0
                value = _decay_to_baseline(float(ema), age_hours)
                return round(max(0.1, min(5.0, value)), 3)

            if profile_id and profile_id in self._data:
                pdata = self._ensure_profile_schema(profile_id)
                bucket_key = _time_bucket(now.hour)
                bucket_entry = pdata.get("bucket_stats", {}).get(bucket_key)
                bucket_value = _from_entry(bucket_entry)
                if bucket_value is not None:
                    return bucket_value

                overall_value = _from_entry(pdata.get("stats"))
                if overall_value is not None:
                    return overall_value

            # Fallback: aggregate across all profiles
            aggregated: list[float] = []
            for pid in list(self._data.keys()):
                pdata = self._ensure_profile_schema(pid)
                value = _from_entry(pdata.get("stats"))
                if value is not None:
                    aggregated.append(value)
            if aggregated:
                return round(sum(aggregated) / len(aggregated), 3)

            # As last resort, fall back to averaging raw samples
            samples: list[tuple[str, float]] = []
            if profile_id and profile_id in self._data:
                samples = self._data[profile_id].get("samples", [])
            else:
                for pdata in self._data.values():
                    samples.extend(pdata.get("samples", []))

            if not samples:
                return 1.0

            weighted_sum = 0.0
            total_weight = 0.0
            for ts, speed in samples[-100:]:
                dt = dt_util.parse_datetime(ts)
                if not dt:
                    continue
                hours_old = (now - dt).total_seconds() / 3600.0
                weight = 0.5 ** (hours_old / (DECAY_HALF_LIFE_HOURS / 2))
                weighted_sum += float(speed) * weight
                total_weight += weight

            if total_weight <= 0:
                return 1.0

            avg = weighted_sum / total_weight
            return round(max(0.1, min(5.0, avg)), 3)
        except Exception:
            _LOGGER.exception("Adaptive avg_speed calculation failed")
            return 1.0


    # -------------------------------------------------------------------------
    # Charging sessions
    # -------------------------------------------------------------------------
    def start_session(self, profile_id: str, level_now: float) -> None:
        """Starte neuen Ladevorgang (wird bei Ladebeginn aufgerufen)."""
        pdata = self._ensure_profile_schema(profile_id)
        pdata["current_session"] = (
            dt_util.now().isoformat(),
            float(level_now),
        )
        _LOGGER.debug("Session started for %s at %.1f%%", profile_id, level_now)

    async def end_session(self, profile_id: str, level_end: float) -> None:
        """Beende aktuellen Ladevorgang."""
        p = self._ensure_profile_schema(profile_id)
        sess = p.pop("current_session", None)
        if not sess:
            _LOGGER.debug("No active session for %s", profile_id)
            return
        try:
            start_time = dt_util.parse_datetime(sess[0]) or dt_util.now()
            start_level = float(sess[1])
            await self.record_cycle(
                profile_id=profile_id,
                start_time=start_time,
                end_time=dt_util.now(),
                start_level=start_level,
                end_level=float(level_end),
                reached_target=start_level < level_end,
                error=None,
            )
        except Exception:
            _LOGGER.exception("Error while ending session for %s", profile_id)

    async def record_cycle(
        self,
        profile_id: str,
        start_time: datetime,
        end_time: datetime,
        start_level: float,
        end_level: float,
        reached_target: bool,
        error: Optional[str],
    ) -> None:
        """Erfasse Ladezyklus und berechne %/min."""
        p = self._ensure_profile_schema(profile_id)
        duration_min = max(0.0, (end_time - start_time).total_seconds() / 60.0)
        try:
            delta = max(0.0, end_level - start_level)
            speed = delta / duration_min if duration_min > 0 else 0.0
        except Exception:
            speed = 0.0

        # Nur plausible Werte speichern
        if speed <= 0 or speed > 10:
            _LOGGER.debug("Ignoring implausible speed %.3f for %s", speed, profile_id)
            return

        timestamp = dt_util.now().isoformat()
        p["samples"].append((timestamp, speed))
        p["cycles"].append(
            {
                "start_time": getattr(start_time, "isoformat", lambda: start_time)(),
                "end_time": getattr(end_time, "isoformat", lambda: end_time)(),
                "start_level": round(start_level, 1),
                "end_level": round(end_level, 1),
                "duration_min": round(duration_min, 1),
                "speed": round(speed, 3),
                "reached_target": reached_target,
                "error_cause": error,
            }
        )

        _LOGGER.debug(
            "Recorded cycle for %s: Δ%.1f%% in %.1fmin (%.3f %%/min)",
            profile_id,
            end_level - start_level,
            duration_min,
            speed,
        )

        self._update_stats(p, speed, timestamp, start_time)
        self._schedule_save()

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------
    def cleanup_old_data(self, max_samples: int = 200, max_cycles: int = 100) -> None:
        """Behalte nur begrenzte Anzahl von Einträgen."""
        for pid, pdata in self._data.items():
            if "samples" in pdata:
                pdata["samples"] = pdata["samples"][-max_samples:]
            if "cycles" in pdata:
                pdata["cycles"] = pdata["cycles"][-max_cycles:]
            if "bucket_stats" in pdata:
                for bucket, entry in list(pdata["bucket_stats"].items()):
                    if not entry.get("ema"):
                        pdata["bucket_stats"].pop(bucket, None)
        _LOGGER.debug("Cleaned up learning data (max %d samples, %d cycles)", max_samples, max_cycles)
        self._schedule_save()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _default_profile(self) -> Dict[str, Any]:
        return {
            "samples": [],
            "cycles": [],
            "stats": {"ema": None, "count": 0, "last_sample": None},
            "bucket_stats": {},
        }

    def _ensure_profile_schema(self, profile_id: str) -> Dict[str, Any]:
        pdata = self._data.setdefault(profile_id, self._default_profile())
        pdata.setdefault("samples", [])
        pdata.setdefault("cycles", [])
        pdata.setdefault("stats", {"ema": None, "count": 0, "last_sample": None})
        pdata.setdefault("bucket_stats", {})
        return pdata

    def _update_stats(self, profile: Dict[str, Any], speed: float, timestamp: str, start_time: datetime) -> None:
        stats = profile.setdefault("stats", {"ema": None, "count": 0, "last_sample": None})
        stats["ema"] = _ema_update(stats.get("ema"), speed)
        stats["count"] = int(stats.get("count", 0)) + 1
        stats["last_sample"] = timestamp

        bucket_map = profile.setdefault("bucket_stats", {})
        local_start = dt_util.as_local(start_time)
        bucket_key = _time_bucket(local_start.hour)
        bucket_entry = bucket_map.setdefault(bucket_key, {"ema": None, "count": 0, "last_sample": None})
        bucket_entry["ema"] = _ema_update(bucket_entry.get("ema"), speed, alpha=EMA_ALPHA)
        bucket_entry["count"] = int(bucket_entry.get("count", 0)) + 1
        bucket_entry["last_sample"] = timestamp

    def _schedule_save(self) -> None:
        if self._save_debounce_unsub:
            self._save_debounce_unsub()

        def _async_save_callback(_now: Any) -> None:
            self._save_debounce_unsub = None
            self.hass.async_create_task(self.async_save())

        self._save_debounce_unsub = async_call_later(
            self.hass,
            SAVE_DEBOUNCE_SECONDS,
            _async_save_callback,
        )
