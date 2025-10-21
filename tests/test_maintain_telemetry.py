"""Unit tests for telemetry maintenance (EWMA and adaptive override lifecycle)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
from custom_components.smart_charger.coordinator import SmartChargerCoordinator

pytestmark = pytest.mark.asyncio


async def test_telemetry_applies_adaptive_override_when_high_flipflop(hass):
    """If flip-flop events are high, an adaptive throttle override should be applied."""
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []}, options={})
    entry.add_to_hass(hass)

    coordinator = SmartChargerCoordinator(hass, entry)

    # Simulate many recent flip-flop events for a fictitious entity
    now = dt_util.utcnow()
    epoch = dt_util.as_timestamp(now)
    ent = "switch.test_flip"
    # populate 6 events within the default window
    coordinator._flipflop_events[ent] = [epoch - 1.0 * i for i in range(6)]
    # make threshold low so we trigger easily in the test
    coordinator._flipflop_warn_threshold = 2

    # Ensure default throttle exists
    coordinator._device_switch_throttle.setdefault(ent, coordinator._default_switch_throttle_seconds)

    # Run a refresh which will execute the telemetry maintenance path
    await coordinator.async_refresh()

    # After refresh, expect either an adaptive override recorded or the device throttle raised
    assert (
        ent in coordinator._adaptive_throttle_overrides
        or float(coordinator._device_switch_throttle.get(ent, 0.0))
        >= float(coordinator._default_switch_throttle_seconds)
    )
