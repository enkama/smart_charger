from datetime import timedelta

from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from custom_components.smart_charger.const import DOMAIN


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

    assert coord._quick_gate_suppress(ent, last_epoch, coord._device_switch_throttle[ent], False) is True


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

    assert coord._quick_gate_suppress(ent, last_epoch, coord._device_switch_throttle[ent], False) is False
