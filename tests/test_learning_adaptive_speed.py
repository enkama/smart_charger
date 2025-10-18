"""Tests for adaptive average speed calculation in Smart Charger learning."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from homeassistant.util import dt as dt_util

from custom_components.smart_charger.const import (
    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    LEARNING_DEFAULT_SPEED,
)
from custom_components.smart_charger.learning import SmartChargerLearning

pytestmark = pytest.mark.asyncio


class _MockHass:
    def __init__(self) -> None:
        self.data = {}
        self.config = SimpleNamespace(config_dir=".")
        self.loop = asyncio.get_event_loop()
        self.state = None

    def async_create_task(self, coro):
        return coro


def _create_learning() -> SmartChargerLearning:
    hass = _MockHass()
    learning = SmartChargerLearning(hass, entry_id="test")
    learning._data = learning._default_storage()  # type: ignore[attr-defined]
    learning._schedule_save = lambda: None  # type: ignore[assignment]
    learning._save_debounce_unsub = None  # type: ignore[attr-defined]
    return learning


async def test_recent_sample_short_circuits_to_fresh_speed() -> None:
    """Most recent sample within the freshness window should dominate the average."""

    learning = _create_learning()
    profile_id = "device"
    now = dt_util.utcnow()

    # Seed slow historic data.
    learning._profiles.setdefault(profile_id, {  # type: ignore[attr-defined]
        "samples": [
            ((now - timedelta(hours=10)).isoformat(), 5.0),
            ((now - timedelta(hours=6)).isoformat(), 6.0),
        ],
        "stats": {
            "ema": 6.0,
            "last_sample": (now - timedelta(hours=6)).isoformat(),
        },
    })

    # Add fast recent sample inside freshness window.
    fresh_speed = 30.0
    learning._profiles[profile_id]["samples"].append(  # type: ignore[index]
        (now.isoformat(), fresh_speed)
    )
    learning._profiles[profile_id]["stats"]["ema"] = 12.0  # type: ignore[index]
    learning._profiles[profile_id]["stats"]["last_sample"] = now.isoformat()  # type: ignore[index]

    result = learning.avg_speed(profile_id)
    assert result == pytest.approx(fresh_speed)


async def test_fallback_to_recent_average_when_sample_outside_window() -> None:
    """When the freshest sample is stale, fall back to the weighted recent average."""

    learning = _create_learning()
    profile_id = "device"
    now = dt_util.utcnow()
    stale_age = timedelta(hours=DEFAULT_LEARNING_RECENT_SAMPLE_HOURS + 1)

    learning._profiles.setdefault(profile_id, {  # type: ignore[attr-defined]
        "samples": [
            ((now - timedelta(hours=10)).isoformat(), 5.0),
            ((now - timedelta(hours=4)).isoformat(), 8.0),
            ((now - stale_age).isoformat(), 20.0),
        ],
        "stats": {
            "ema": 9.0,
            "last_sample": (now - stale_age).isoformat(),
        },
    })

    result = learning.avg_speed(profile_id)
    assert result > LEARNING_DEFAULT_SPEED
    assert result < 20.0


async def test_cycle_outlier_rejection_respects_high_speed_cap() -> None:
    """Ensure record_cycle accepts higher speeds when history is sparse."""

    learning = _create_learning()
    profile_id = "device"
    now = dt_util.utcnow()

    await learning.ensure_profile(profile_id)
    learning._profiles[profile_id]["stats"].update({  # type: ignore[index]
        "ema": 8.0,
        "count": 2,
    })

    accepted = await learning.record_cycle(
        profile_id=profile_id,
        start_time=now - timedelta(minutes=20),
        end_time=now,
        start_level=40,
        end_level=60,
        reached_target=True,
        error=None,
    )

    assert accepted
    speeds = [speed for _, speed in learning._profiles[profile_id]["samples"]]  # type: ignore[index]
    assert pytest.approx(speeds[-1], rel=1e-3) == 60.0