"""Test that sustained EWMA exceed causes adaptive_mode_override to become 'aggressive'."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.const import DOMAIN

pytestmark = pytest.mark.asyncio


async def test_sustained_ewma_triggers_aggressive(hass) -> None:
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    # Ensure initial state
    assert coordinator._adaptive_mode_override is None

    # Simulate many flipflop events in the window to create high EWMA
    ent = "switch.testcharger"
    now = dt_util.as_timestamp(dt_util.utcnow())
    # push events to coordinator._flipflop_events (timestamps in epoch seconds)
    # create enough events so rate > threshold (warn_threshold/window)
    warn = getattr(coordinator, "_flipflop_warn_threshold", 3)

    # add warn + 2 events to ensure excess
    timestamps = [now - 1.0 * i for i in range(warn + 2)]
    coordinator._flipflop_events[ent] = timestamps

    # Run coordinator update to compute EWMA first time
    await coordinator.async_refresh()

    # Now simulate sustained exceed over 5 minutes: advance the "since" timestamp
    # Set exceeded_since to 5 minutes ago so next update will detect duration >= 300s
    coordinator._flipflop_ewma_exceeded_since = dt_util.as_timestamp(dt_util.utcnow() - timedelta(seconds=301))
    coordinator._flipflop_ewma = 1.0
    coordinator._flipflop_ewma_exceeded = True

    # Trigger another refresh which should apply the aggressive override
    await coordinator.async_refresh()

    assert coordinator._adaptive_mode_override == "aggressive"
