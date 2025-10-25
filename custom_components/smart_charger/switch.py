from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.device_registry import DeviceInfo

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up per-device auto-control Switch entities."""
    domain_data = hass.data[DOMAIN]
    data = domain_data["entries"][entry.entry_id]
    coordinator = data["coordinator"]

    devices = entry.data.get("devices", [])
    entities: list[SwitchEntity] = []
    for device in devices:
        name = device.get("name")
        if name:
            entities.append(SmartChargerAutoControlSwitch(name, entry, coordinator))

    if entities:
        async_add_entities(entities, True)


class SmartChargerAutoControlSwitch(SwitchEntity):
    """Per-device toggle to enable/disable automatic control.

    The state is persisted in the config entry options under the key
    `device_auto_control_enabled` as a mapping device_name -> bool.
    """

    _attr_should_poll = False

    def __init__(self, device_name: str, entry, coordinator) -> None:
        self.device_name = device_name
        self.entry = entry
        self.coordinator = coordinator
        dn = device_name.lower().replace(" ", "_")
        self._attr_name = f"Smart Charger {device_name} Auto Control"
        self._attr_unique_id = f"{DOMAIN}_{dn}_auto_control"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, dn)},
            name=f"Smart Charger – {device_name}",
            manufacturer="Smart Charger System",
            model="Predictive Charging v2",
            configuration_url=("https://my.home-assistant.io/redirect/integrations/"),
        )
        # initialize cached attribute used by the base SwitchEntity
        try:
            opts = dict(self.entry.options or {})
            mapping = opts.get("device_auto_control_enabled") or {}
            self._attr_is_on = bool(mapping.get(self.device_name, True))
        except Exception:
            _LOGGER.debug(
                "Failed to read initial auto-control option, defaulting to enabled",
                exc_info=True,
            )
            self._attr_is_on = True

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self._update_option(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._update_option(False)

    async def _update_option(self, val: bool) -> None:
        # Read current options and update the per-device mapping
        opts = dict(self.entry.options or {})
        mapping = dict(opts.get("device_auto_control_enabled") or {})
        mapping[self.device_name] = bool(val)
        opts["device_auto_control_enabled"] = mapping
        # Persist options on the config entry
        try:
            await self.entry.async_update_options(opts)
        except Exception:
            # Best-effort: update in-memory options so UI reacts until persisted
            try:
                self.entry.options = opts
            except Exception:
                _LOGGER.debug(
                    "Failed to persist options to entry; keeping in-memory",
                    exc_info=True,
                )
        # Trigger coordinator update so other entities pick up change
        try:
            if hasattr(self.coordinator, "async_request_refresh"):
                await self.coordinator.async_request_refresh()
        except Exception:
            _LOGGER.debug(
                "Failed to request coordinator refresh after option change",
                exc_info=True,
            )
        # update cached attribute and write state so the UI updates immediately
        try:
            self._attr_is_on = bool(val)
            try:
                # async_write_ha_state is provided by Entity base class
                self.async_write_ha_state()
            except Exception:
                _LOGGER.debug(
                    "Failed to write HA state after option change", exc_info=True
                )
        except Exception:
            _LOGGER.debug("Failed to update cached _attr_is_on", exc_info=True)
