import time
from datetime import datetime

from custom_components.smart_charger.coordinator import SmartChargerCoordinator
from pytest_homeassistant_custom_component.common import MockConfigEntry
from custom_components.smart_charger.const import DOMAIN


def test_expire_adaptive_overrides(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"devices": []})
    entry.add_to_hass(hass)

    coord = SmartChargerCoordinator(hass, entry)

    ent = "switch.test"
    # set a device throttle to an original value
    coord._device_switch_throttle[ent] = 5.0

    # create an override that already expired
    now = time.time()
    coord._adaptive_throttle_overrides[ent] = {"original": 5.0, "applied": 10.0, "expires": now - 10}

    # call helper
    coord._expire_adaptive_overrides(now)

    # override should be removed
    assert ent not in coord._adaptive_throttle_overrides
    # device throttle should be restored
    assert coord._device_switch_throttle[ent] == 5.0
