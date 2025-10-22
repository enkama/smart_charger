import pytest
import time
from typing import Any

from custom_components.smart_charger.coordinator import SmartChargerCoordinator


class DummyHass:
    pass


@pytest.mark.asyncio
async def test_confirmation_counts_and_throttle(tmp_path: Any):
    hass = DummyHass()
    coordinator = SmartChargerCoordinator(hass, None)  # type: ignore

    # set defaults
    coordinator._confirmation_required = 3
    coordinator._default_switch_throttle_seconds = 60

    norm = "switch.test_device"

    # ensure no prior history
    coordinator._desired_state_history.pop(norm, None)
    coordinator._last_switch_time.pop(norm, None)

    # first call should record and suppress (count=1 < required=3)
    suppress, hist, required, count = coordinator._confirmation_and_throttle_check(
        norm, True
    )
    assert suppress is True
    assert required == 3
    assert count == 1
    assert hist[0] is True

    # second call within same flow: simulate new eval by clearing _last_recorded_eval
    coordinator._last_recorded_eval.pop(norm, None)
    suppress, hist, required, count = coordinator._confirmation_and_throttle_check(
        norm, True
    )
    assert suppress is True
    assert count == 2

    # third call reaches required and should not be suppressed
    coordinator._last_recorded_eval.pop(norm, None)
    suppress, hist, required, count = coordinator._confirmation_and_throttle_check(
        norm, True
    )
    assert suppress is False
    assert count == 3


@pytest.mark.asyncio
async def test_throttle_prevents_action(tmp_path: Any):
    hass = DummyHass()
    coordinator = SmartChargerCoordinator(hass, None)  # type: ignore

    coordinator._confirmation_required = 1
    coordinator._default_switch_throttle_seconds = 3600

    norm = "switch.test_device2"
    coordinator._desired_state_history.pop(norm, None)

    # Use numeric timestamp for last switch (now) so throttle should block
    coordinator._last_switch_time[norm] = time.time()

    # ensure we don't bypass throttle
    suppress, hist, required, count = coordinator._confirmation_and_throttle_check(
        norm, True
    )
    assert suppress is True
