from __future__ import annotations

from typing import Any, Callable
import logging

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers import device_registry as dr
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    PLATFORMS,
    SERVICE_FORCE_REFRESH,
    SERVICE_START_CHARGING,
    SERVICE_STOP_CHARGING,
    SERVICE_AUTO_MANAGE,
    SERVICE_LOAD_MODEL,
    CONF_BATTERY_SENSOR,
    CONF_CHARGING_SENSOR,
)
from .coordinator import SmartChargerCoordinator
from .services import (
    SmartChargerStateMachine,
    handle_force_refresh,
    handle_start_charging,
    handle_stop_charging,
    handle_auto_manage,
    handle_load_model,
)
from .learning import SmartChargerLearning

_LOGGER = logging.getLogger(__name__)


def _get_domain_data(hass: HomeAssistant) -> dict[str, Any]:
    data = hass.data.setdefault(DOMAIN, {})
    data.setdefault("entries", {})
    data.setdefault("services_registered", False)
    return data


def _resolve_entry_context(hass: HomeAssistant, call: ServiceCall) -> tuple[ConfigEntry, dict[str, Any]]:
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

    raise HomeAssistantError("Specify 'entry_id' in service data when multiple Smart Charger entries are configured")


def _register_services(hass: HomeAssistant) -> None:
    domain_data = _get_domain_data(hass)

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

    for name, func in (
        (SERVICE_FORCE_REFRESH, _svc_force_refresh),
        (SERVICE_START_CHARGING, _svc_start),
        (SERVICE_STOP_CHARGING, _svc_stop),
        (SERVICE_AUTO_MANAGE, _svc_auto),
        (SERVICE_LOAD_MODEL, _svc_load_model),
    ):
        if not hass.services.has_service(DOMAIN, name):
            hass.services.async_register(DOMAIN, name, func)

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

    # Starten, damit der Coordinator auch ohne Entities zyklisch aktualisiert
    polling_unsub = None
    start_polling = getattr(coordinator, "async_start_polling", None)
    if callable(start_polling):
        polling_unsub = start_polling()
    entry_data["coordinator_polling_unsub"] = polling_unsub

    state_machine = SmartChargerStateMachine(hass)
    await state_machine.async_load()

    learning = SmartChargerLearning(hass)
    await learning.async_load()

    devices = entry.data.get("devices") or []

    entry_data.update(
        {
            "state_machine": state_machine,
            "learning": learning,
            "unsub_listeners": [],
        }
    )

    # ---------------- Geräte-Registrierung ----------------
    device_registry = dr.async_get(hass)

    # Hauptgerät für globale Sensoren
    device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, "main_hub")},
        manufacturer="Smart Charger System",
        name="Smart Charger",
        model="Predictive Charging Hub",
        configuration_url="https://my.home-assistant.io/redirect/integrations/",
    )

    # Untergeräte für jedes konfigurierte Device
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

    # --- Load platforms (wichtig, damit Sensoren sichtbar werden) ---
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception as err:
        _LOGGER.exception("Error forwarding platforms: %s", err)

    # ---------------- Entity updates ----------------
    async def _on_entity_change(event: Any) -> None:
        """Recalculate charging plan when relevant sensor changes."""
        data = entries.get(entry.entry_id)
        if not data:
            return
        cfg = {**entry.data, **getattr(entry, "options", {})}
        await handle_auto_manage(
            hass,
            entry.entry_id,
            cfg,
            data["coordinator"],
            data["state_machine"],
            data["learning"],
        )

    listeners: list[Callable[[], None]] = entry_data["unsub_listeners"]

    for device in devices:
        for key in (CONF_BATTERY_SENSOR, CONF_CHARGING_SENSOR):
            ent = device.get(key)
            if ent:
                listeners.append(async_track_state_change_event(hass, ent, _on_entity_change))

        for key in (
            "alarm_entity",
            "alarm_entity_monday",
            "alarm_entity_tuesday",
            "alarm_entity_wednesday",
            "alarm_entity_thursday",
            "alarm_entity_friday",
            "alarm_entity_saturday",
            "alarm_entity_sunday",
        ):
            ent = device.get(key)
            if ent:
                listeners.append(async_track_state_change_event(hass, ent, _on_entity_change))

    # ---------------- Presence-triggered auto-refresh ----------------
    presence_entities = [d.get("presence_sensor") for d in devices if d.get("presence_sensor")]

    async def _on_presence_change(event: Any) -> None:
        new_state = event.data.get("new_state")
        if not new_state or new_state.entity_id not in presence_entities:
            return
        if str(new_state.state).lower() in ("home", "on", "present", "true"):
            _LOGGER.debug("Smart Charger: %s became home → triggering refresh", new_state.entity_id)
            refresh = getattr(coordinator, "async_throttled_refresh", None)
            if callable(refresh):
                await refresh()  # type: ignore[func-returns-value]
            else:
                await coordinator.async_request_refresh()

    for ent in presence_entities:
        listeners.append(async_track_state_change_event(hass, ent, _on_presence_change))

    entry_data["update_listener_unsub"] = entry.add_update_listener(_async_reload_entry)

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

