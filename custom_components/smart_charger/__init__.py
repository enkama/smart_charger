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
from homeassistant.util import dt as dt_util

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


async def _svc_force_refresh(call: ServiceCall) -> None:
    """Service: force refresh a coordinator for an entry."""
    _, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    await handle_force_refresh(call.hass, entry_data["coordinator"])


async def _svc_start(call: ServiceCall) -> None:
    """Service: start charging for an entry/state machine."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    cfg = {**entry.data, **getattr(entry, "options", {})}
    await handle_start_charging(call.hass, cfg, call, entry_data["state_machine"])


async def _svc_stop(call: ServiceCall) -> None:
    """Service: stop charging for an entry/state machine."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    cfg = {**entry.data, **getattr(entry, "options", {})}
    await handle_stop_charging(call.hass, cfg, call, entry_data["state_machine"])


async def _svc_auto(call: ServiceCall) -> None:
    """Service: toggle auto manage for an entry."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    cfg = {**entry.data, **getattr(entry, "options", {})}
    await handle_auto_manage(
        call.hass,
        entry.entry_id,
        cfg,
        entry_data["coordinator"],
        entry_data["state_machine"],
        entry_data["learning"],
    )


async def _svc_load_model(call: ServiceCall) -> None:
    """Service: load model into learning component."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    cfg = {**entry.data, **getattr(entry, "options", {})}
    await handle_load_model(call.hass, cfg, call, entry_data["learning"])


async def _svc_set_adaptive_override(call: ServiceCall) -> None:
    """Service: set entry-level adaptive mode override and persist it."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    mode = str(call.data.get("mode", "")).strip().lower() or None
    if mode not in ("conservative", "normal", "aggressive"):
        raise HomeAssistantError("Invalid adaptive mode")
    coordinator: SmartChargerCoordinator = entry_data["coordinator"]
    coordinator._adaptive_mode_override = mode
    # persist to entry options
    try:
        new_opts = dict(getattr(entry, "options", {}) or {})
        new_opts["adaptive_mode_override"] = mode
        try:
            call.hass.config_entries.async_update_entry(entry, options=new_opts)
        except Exception:
            _LOGGER.debug("Failed to persist adaptive override to config entry options", exc_info=True)
    except Exception:
        _LOGGER.debug("Failed to persist adaptive override (unexpected)")


async def _svc_clear_adaptive_override(call: ServiceCall) -> None:
    """Service: clear entry-level adaptive mode override and remove persistence."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    coordinator: SmartChargerCoordinator = entry_data["coordinator"]
    coordinator._adaptive_mode_override = None
    # remove persisted key if present
    try:
        new_opts = dict(getattr(entry, "options", {}) or {})
        if "adaptive_mode_override" in new_opts:
            new_opts.pop("adaptive_mode_override", None)
            try:
                call.hass.config_entries.async_update_entry(entry, options=new_opts)
            except Exception:
                _LOGGER.debug("Failed to clear adaptive override from config entry options", exc_info=True)
    except Exception:
        _LOGGER.debug("Failed to clear adaptive override (unexpected)")


async def _svc_set_adaptive_override_entity(call: ServiceCall) -> None:
    """Service: set per-entity adaptive override and persist mapping."""
    entry, entry_data = _resolve_entry_context(hass=call.hass, call=call)
    entity_id = call.data.get("entity_id")
    mode = str(call.data.get("mode", "")).strip().lower() or None
    if not entity_id:
        raise HomeAssistantError("entity_id is required")
    if mode not in ("conservative", "normal", "aggressive"):
        raise HomeAssistantError("Invalid adaptive mode")
    coordinator: SmartChargerCoordinator = entry_data["coordinator"]
    # apply per-entity throttle override
    try:
        overrides = getattr(coordinator, "_adaptive_throttle_overrides", {}) or {}
        overrides[entity_id] = {"mode": mode, "applied": True}
        coordinator._adaptive_throttle_overrides = overrides
    except Exception:
        _LOGGER.debug("Failed to apply in-memory entity override")
    # persist mapping in entry options
    try:
        new_opts = dict(getattr(entry, "options", {}) or {})
        mapping = dict(new_opts.get("adaptive_mode_overrides", {}) or {})
        mapping[entity_id] = mode
        new_opts["adaptive_mode_overrides"] = mapping
        try:
            call.hass.config_entries.async_update_entry(entry, options=new_opts)
        except Exception:
            _LOGGER.debug("Failed to persist entity override to config entry options", exc_info=True)
    except Exception:
        _LOGGER.debug("Failed to persist entity override (unexpected)")
 

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

    override_set_schema = base_schema.extend(
        {
            vol.Required("mode"): vol.In({"conservative", "normal", "aggressive"}),
        }
    )

    override_clear_schema = base_schema

    # Prepare entity-level schema (defined here so it's available in the registration tuple)
    entity_override_schema = base_schema.extend(
        {
            vol.Required("entity_id"): vol.Any(cv.entity_id, cv.entity_ids),
            vol.Required("mode"): vol.In({"conservative", "normal", "aggressive"}),
        }
    )

    for name, func, schema in (
        (SERVICE_FORCE_REFRESH, _svc_force_refresh, base_schema),
        (SERVICE_START_CHARGING, _svc_start, entity_schema),
        (SERVICE_STOP_CHARGING, _svc_stop, entity_schema),
        (SERVICE_AUTO_MANAGE, _svc_auto, base_schema),
        (SERVICE_LOAD_MODEL, _svc_load_model, load_model_schema),
        ("set_adaptive_override", _svc_set_adaptive_override, override_set_schema),
        ("clear_adaptive_override", _svc_clear_adaptive_override, override_clear_schema),
        ("set_adaptive_override_entity", _svc_set_adaptive_override_entity, entity_override_schema),
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
    # After the coordinator has loaded its options, translate any persisted
    # per-entity mode overrides into numeric throttle overrides so they
    # immediately affect runtime behavior.
    try:
        persisted = dict(getattr(entry, "options", {}) or {}).get("adaptive_mode_overrides") or {}
        if isinstance(persisted, dict) and persisted:
            now_epoch = float(dt_util.as_timestamp(dt_util.utcnow()))
            overrides = getattr(coordinator, "_adaptive_throttle_overrides", {}) or {}
            for ent_id, mode_raw in persisted.items():
                try:
                    mode = str(mode_raw).strip().lower()
                    # Determine factor mapping consistent with coordinator mode factors
                    if mode == "conservative":
                        mode_factor = 0.7
                    elif mode == "aggressive":
                        mode_factor = 1.4
                    else:
                        mode_factor = 1.0

                    # Current configured throttle for entity (seconds)
                    current = float(coordinator._device_switch_throttle.get(ent_id, coordinator._default_switch_throttle_seconds) or coordinator._default_switch_throttle_seconds)
                    # Compute a starting multiplier using coordinator defaults/backoff base
                    var_multiplier = float(getattr(coordinator, "_adaptive_throttle_multiplier", 1.0)) * float(mode_factor)
                    # Clamp and compute desired applied throttle
                    desired = max(current * var_multiplier, float(getattr(coordinator, "_adaptive_throttle_min_seconds", 0.0)))
                    expires = float(now_epoch + float(getattr(coordinator, "_adaptive_throttle_duration_seconds", 600.0)))
                    overrides[ent_id] = {
                        "original": float(current),
                        "applied": float(desired),
                        "expires": float(expires),
                    }
                    # Apply to runtime throttle map so coordinator uses it immediately
                    coordinator._device_switch_throttle[ent_id] = float(desired)
                except Exception:
                    _LOGGER.debug("Ignoring malformed persisted entity override for %s", ent_id)
            coordinator._adaptive_throttle_overrides = overrides
    except Exception:
        _LOGGER.debug("Failed to load persisted per-entity overrides from entry.options")

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
