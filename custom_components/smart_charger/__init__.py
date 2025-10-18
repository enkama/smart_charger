from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.event import async_track_state_change_event

from .const import (
    CONF_BATTERY_SENSOR,
    CONF_CHARGING_SENSOR,
    CONF_PRESENCE_SENSOR,
    DOMAIN,
    PLATFORMS,
    SERVICE_AUTO_MANAGE,
    SERVICE_FORCE_REFRESH,
    SERVICE_LOAD_MODEL,
    SERVICE_START_CHARGING,
    SERVICE_STOP_CHARGING,
)
from .coordinator import SmartChargerCoordinator
from .learning import SmartChargerLearning
from .services import (
    SmartChargerStateMachine,
    handle_auto_manage,
    handle_force_refresh,
    handle_load_model,
    handle_start_charging,
    handle_stop_charging,
)

_LOGGER = logging.getLogger(__name__)


def _get_domain_data(hass: HomeAssistant) -> dict[str, Any]:
    data = hass.data.setdefault(DOMAIN, {})
    data.setdefault("entries", {})
    data.setdefault("services_registered", False)
    return data


DEVICE_SENSOR_KEYS: tuple[str, ...] = (CONF_BATTERY_SENSOR, CONF_CHARGING_SENSOR)
ALARM_ENTITY_KEYS: tuple[str, ...] = (
    "alarm_entity",
    "alarm_entity_monday",
    "alarm_entity_tuesday",
    "alarm_entity_wednesday",
    "alarm_entity_thursday",
    "alarm_entity_friday",
    "alarm_entity_saturday",
    "alarm_entity_sunday",
)
PRESENCE_ACTIVE_STATES: set[str] = {"home", "on", "present", "true"}


def _merged_entry_config(entry: ConfigEntry) -> dict[str, Any]:
    options = getattr(entry, "options", {}) or {}
    return {**entry.data, **options}


def _maybe_start_coordinator_polling(coordinator: SmartChargerCoordinator) -> Any:
    start_polling = getattr(coordinator, "async_start_polling", None)
    if callable(start_polling):
        return start_polling()
    return None


def _create_auto_manage_debouncer(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator: SmartChargerCoordinator,
    state_machine: SmartChargerStateMachine,
    learning: SmartChargerLearning,
) -> Debouncer:
    async def _async_auto_manage() -> None:
        cfg = _merged_entry_config(entry)
        await handle_auto_manage(
            hass,
            entry.entry_id,
            cfg,
            coordinator,
            state_machine,
            learning,
        )

    return Debouncer(
        hass,
        _LOGGER,
        cooldown=2.0,
        immediate=False,
        function=_async_auto_manage,
    )


def _register_devices_with_registry(
    hass: HomeAssistant, entry: ConfigEntry, devices: list[dict[str, Any]]
) -> None:
    device_registry = dr.async_get(hass)
    for device in devices:
        name = device.get("name")
        if not name:
            continue
        device_registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, name.lower().replace(" ", "_"))},
            manufacturer="Smart Charger System",
            name=f"Smart Charger – {name}",
            model="Predictive Charging v2",
            configuration_url="https://my.home-assistant.io/redirect/integrations/",
        )


async def _safe_forward_entry_setups(hass: HomeAssistant, entry: ConfigEntry) -> None:
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception as err:
        _LOGGER.exception("Error forwarding platforms: %s", err)


def _attach_device_state_listeners(
    hass: HomeAssistant,
    entry_data: dict[str, Any],
    devices: list[dict[str, Any]],
    callback: Callable[[Any], Any],
) -> list[str]:
    listeners: list[Callable[[], None]] = entry_data.setdefault("unsub_listeners", [])
    for device in devices:
        for key in DEVICE_SENSOR_KEYS:
            ent = device.get(key)
            if ent:
                listeners.append(async_track_state_change_event(hass, ent, callback))
        for key in ALARM_ENTITY_KEYS:
            ent = device.get(key)
            if ent:
                listeners.append(async_track_state_change_event(hass, ent, callback))
    return [
        device[CONF_PRESENCE_SENSOR]
        for device in devices
        if device.get(CONF_PRESENCE_SENSOR)
    ]


def _attach_presence_listeners(
    hass: HomeAssistant,
    entry_data: dict[str, Any],
    presence_entities: list[str],
    callback: Callable[[Any], Any],
) -> None:
    listeners: list[Callable[[], None]] = entry_data.setdefault("unsub_listeners", [])
    for ent in presence_entities:
        listeners.append(async_track_state_change_event(hass, ent, callback))


def _make_entity_change_callback(
    hass: HomeAssistant,
    entries: dict[str, dict[str, Any]],
    entry_id: str,
) -> Callable[[Any], Any]:
    async def _on_entity_change(event: Any) -> None:
        data = entries.get(entry_id)
        if not data:
            return
        debouncer: Debouncer | None = data.get("auto_manage_debouncer")
        if debouncer:
            hass.async_create_task(debouncer.async_call())

    return _on_entity_change


def _make_presence_callback(
    hass: HomeAssistant,
    entry_data: dict[str, Any],
    coordinator: SmartChargerCoordinator,
    presence_entities: list[str],
) -> Callable[[Any], Any]:
    presence_targets = {ent for ent in presence_entities if ent}

    async def _on_presence_change(event: Any) -> None:
        new_state = event.data.get("new_state")
        if not new_state or new_state.entity_id not in presence_targets:
            return
        if str(new_state.state).lower() not in PRESENCE_ACTIVE_STATES:
            return
        _LOGGER.debug(
            "Smart Charger: %s became home -> triggering refresh",
            new_state.entity_id,
        )
        debouncer: Debouncer | None = entry_data.get("auto_manage_debouncer")
        if debouncer:
            hass.async_create_task(debouncer.async_call())
            return
        refresh = getattr(coordinator, "async_throttled_refresh", None)
        if callable(refresh):
            result = refresh()
            if inspect.isawaitable(result):
                await result
            return
        await coordinator.async_request_refresh()

    return _on_presence_change


def _trigger_initial_auto_manage(
    hass: HomeAssistant, entry_data: dict[str, Any]
) -> None:
    debouncer: Debouncer | None = entry_data.get("auto_manage_debouncer")
    if debouncer:
        hass.async_create_task(debouncer.async_call())


def _resolve_entry_context(
    hass: HomeAssistant, call: ServiceCall
) -> tuple[ConfigEntry, dict[str, Any]]:
    domain_data = _get_domain_data(hass)
    entries: dict[str, dict[str, Any]] = domain_data["entries"]
    entry_id = call.data.get("entry_id")

    if entry_id:
        entry_data = entries.get(entry_id)
        if not entry_data:
            raise HomeAssistantError(f"Smart Charger entry_id '{entry_id}' not found")
        entry = entry_data.get("entry")
        if not entry:
            raise HomeAssistantError(f"Entry context missing for '{entry_id}'")
        return entry, entry_data

    if len(entries) == 1:
        entry_id, entry_data = next(iter(entries.items()))
        entry = entry_data.get("entry")
        if not entry:
            raise HomeAssistantError(f"Entry context missing for '{entry_id}'")
        return entry, entry_data

    raise HomeAssistantError(
        "Specify 'entry_id' in service data when multiple Smart Charger entries are configured"
    )


def _register_services(hass: HomeAssistant) -> None:
    domain_data = _get_domain_data(hass)

    base_schema = vol.Schema({vol.Optional("entry_id"): cv.string})
    entity_schema = base_schema.extend(
        {
            vol.Optional("entity_id"): vol.Any(cv.entity_id, cv.entity_ids),
        }
    )
    load_model_schema = base_schema.extend(
        {
            vol.Optional("action", default="load"): vol.In({"load", "reset"}),
            vol.Optional("profile_id"): cv.string,
        }
    )

    async def _svc_force_refresh(call: ServiceCall) -> None:
        _, entry_data = _resolve_entry_context(hass, call)
        await handle_force_refresh(hass, entry_data["coordinator"])

    async def _svc_start(call: ServiceCall) -> None:
        entry, entry_data = _resolve_entry_context(hass, call)
        cfg = {**entry.data, **getattr(entry, "options", {})}
        await handle_start_charging(hass, cfg, call, entry_data["state_machine"])

    async def _svc_stop(call: ServiceCall) -> None:
        entry, entry_data = _resolve_entry_context(hass, call)
        cfg = {**entry.data, **getattr(entry, "options", {})}
        await handle_stop_charging(hass, cfg, call, entry_data["state_machine"])

    async def _svc_auto(call: ServiceCall) -> None:
        entry, entry_data = _resolve_entry_context(hass, call)
        cfg = {**entry.data, **getattr(entry, "options", {})}
        await handle_auto_manage(
            hass,
            entry.entry_id,
            cfg,
            entry_data["coordinator"],
            entry_data["state_machine"],
            entry_data["learning"],
        )

    async def _svc_load_model(call: ServiceCall) -> None:
        entry, entry_data = _resolve_entry_context(hass, call)
        cfg = {**entry.data, **getattr(entry, "options", {})}
        await handle_load_model(hass, cfg, call, entry_data["learning"])

    for name, func, schema in (
        (SERVICE_FORCE_REFRESH, _svc_force_refresh, base_schema),
        (SERVICE_START_CHARGING, _svc_start, entity_schema),
        (SERVICE_STOP_CHARGING, _svc_stop, entity_schema),
        (SERVICE_AUTO_MANAGE, _svc_auto, base_schema),
        (SERVICE_LOAD_MODEL, _svc_load_model, load_model_schema),
    ):
        if not hass.services.has_service(DOMAIN, name):
            hass.services.async_register(DOMAIN, name, func, schema=schema)

    domain_data["services_registered"] = True


async def _async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Setup Smart Charger integration."""
    domain_data = _get_domain_data(hass)
    entries: dict[str, dict[str, Any]] = domain_data["entries"]
    entry_data: dict[str, Any] = entries.setdefault(entry.entry_id, {})
    entry_data["entry"] = entry

    if not domain_data["services_registered"]:
        _register_services(hass)

    coordinator = SmartChargerCoordinator(hass, entry)
    entry_data["coordinator"] = coordinator
    await coordinator.async_config_entry_first_refresh()

    entry_data["coordinator_polling_unsub"] = _maybe_start_coordinator_polling(
        coordinator
    )

    state_machine = SmartChargerStateMachine(hass)
    await state_machine.async_load()

    learning = SmartChargerLearning(hass, entry.entry_id)
    await learning.async_load()

    devices = entry.data.get("devices") or []

    entry_data.update(
        {
            "state_machine": state_machine,
            "learning": learning,
            "unsub_listeners": [],
            "auto_manage_debouncer": _create_auto_manage_debouncer(
                hass,
                entry,
                coordinator,
                state_machine,
                learning,
            ),
        }
    )

    _register_devices_with_registry(hass, entry, devices)
    await _safe_forward_entry_setups(hass, entry)

    entity_callback = _make_entity_change_callback(hass, entries, entry.entry_id)
    presence_entities = _attach_device_state_listeners(
        hass, entry_data, devices, entity_callback
    )
    presence_callback = _make_presence_callback(
        hass, entry_data, coordinator, presence_entities
    )
    _attach_presence_listeners(
        hass,
        entry_data,
        presence_entities,
        presence_callback,
    )

    entry_data["update_listener_unsub"] = entry.add_update_listener(_async_reload_entry)

    _trigger_initial_auto_manage(hass, entry_data)

    _LOGGER.info("Smart Charger initialized with entry_id=%s", entry.entry_id)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Smart Charger integration."""
    domain_data = _get_domain_data(hass)
    entries: dict[str, dict[str, Any]] = domain_data["entries"]
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        data = entries.pop(entry.entry_id, {})
        if not entries:
            for svc in (
                SERVICE_FORCE_REFRESH,
                SERVICE_START_CHARGING,
                SERVICE_STOP_CHARGING,
                SERVICE_AUTO_MANAGE,
                SERVICE_LOAD_MODEL,
            ):
                hass.services.async_remove(DOMAIN, svc)
            domain_data["services_registered"] = False
            hass.data.pop(DOMAIN, None)
        polling_unsub = data.get("coordinator_polling_unsub")
        if callable(polling_unsub):
            polling_unsub()
        if update_unsub := data.get("update_listener_unsub"):
            update_unsub()
        for unsub in data.get("unsub_listeners", []):
            try:
                unsub()
            except Exception as err:
                _LOGGER.warning("Error while unsubscribing listener: %s", err)
        if debouncer := data.get("auto_manage_debouncer"):
            try:
                debouncer.async_cancel()
            except Exception as err:  # best effort
                _LOGGER.debug(
                    "Smart Charger debouncer cancel failed for %s: %s",
                    entry.entry_id,
                    err,
                )
        _LOGGER.info("Smart Charger unloaded: %s", entry.entry_id)
    device_registry = dr.async_get(hass)
    devices = entry.data.get("devices", [])

    for dev in devices:
        name = dev.get("name")
        if not name:
            continue
        identifier = (DOMAIN, name.lower().replace(" ", "_"))

        if device_entry := device_registry.async_get_device({identifier}):
            _LOGGER.info("Removing Smart Charger device: %s", name)
            device_registry.async_remove_device(device_entry.id)
    return unload_ok
