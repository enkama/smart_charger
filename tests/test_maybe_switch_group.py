from datetime import timedelta
import time
from typing import Any

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.const import DOMAIN
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


async def test_quick_gate_suppresses_when_recent_opposite(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    ent = "switch.test"
    # last action recorded as True (on)
    coord._last_action_state[ent] = True
    # last epoch 5 seconds ago
    now = dt_util.utcnow()
    coord._current_eval_time = now
    last_epoch = float(dt_util.as_timestamp(now - timedelta(seconds=5)))

    # throttle set to 10 seconds — elapsed (5) < throttle (10) and last action
    # differs from desired (False) so suppression expected
    coord._device_switch_throttle[ent] = 10.0

    assert (
        coord._quick_gate_suppress(
            ent, last_epoch, coord._device_switch_throttle[ent], False
        )
        is True
    )


async def test_quick_gate_allows_when_elapsed_exceeds_throttle(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)
    coord = SmartChargerCoordinator(hass, entry)

    ent = "switch.test"
    coord._last_action_state[ent] = True
    now = dt_util.utcnow()
    coord._current_eval_time = now
    # last epoch 30 seconds ago
    last_epoch = float(dt_util.as_timestamp(now - timedelta(seconds=30)))

    coord._device_switch_throttle[ent] = 5.0

    assert (
        coord._quick_gate_suppress(
            ent, last_epoch, coord._device_switch_throttle[ent], False
        )
        is False
    )
