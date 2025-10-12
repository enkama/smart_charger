from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.const import STATE_ON

from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_CHARGING_SENSOR,
    CONF_TARGET_LEVEL,
    UNKNOWN_STATES,
)

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Machine (für Auto-Management / Status-Übergänge)
# ---------------------------------------------------------------------------

class SmartChargerStateMachine:
    """Einfache interne Zustandsverwaltung für Ladezyklen."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self.states: Dict[str, str] = {}  # profile_id → state
        self.error_history: Dict[str, list[str]] = {}
        self.error_message: Optional[str] = None

    async def async_load(self) -> None:
        """Lädt initiale Zustände."""
        _LOGGER.debug("SmartChargerStateMachine initialized")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "states": dict(self.states),
            "last_error": self.error_message,
        }

    def set_state(self, profile_id: str, state: str) -> None:
        """Aktualisiere Zustand (idle/charging/completed/error)."""
        old = self.states.get(profile_id)
        self.states[profile_id] = state
        if old != state:
            _LOGGER.debug("State for %s changed: %s → %s", profile_id, old, state)

    def add_error(self, profile_id: str, error: str) -> None:
        """Füge Fehler in Verlauf hinzu."""
        ts = dt_util.now().isoformat()
        key = f"{profile_id}:{error}"
        self.error_history.setdefault(key, []).append(ts)
        self.error_message = f"[{profile_id}] {error}"
        _LOGGER.warning("SmartCharger error: %s", self.error_message)

    def get_suggestions(self, profile_id: Optional[str] = None):
        """Liefert gespeicherte Fehler oder Hinweise."""
        if profile_id:
            return [
                key.split(":", 1)[1]
                for key in self.error_history.keys()
                if key.startswith(f"{profile_id}:")
            ]
        return [key.split(":", 1)[1] for key in self.error_history.keys()]


# ---------------------------------------------------------------------------
# Helper: Sensor & State Retrieval
# ---------------------------------------------------------------------------

def _get_state(hass: HomeAssistant, entity_id: Optional[str]) -> Optional[str]:
    if not entity_id:
        return None
    state = hass.states.get(entity_id)
    return state.state if state and state.state not in UNKNOWN_STATES else None


def _is_charging_state(value: Optional[str]) -> bool:
    if not value:
        return False
    return str(value).lower() in ("on", "charging", "true", "1")


def _normalize_entity_ids(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        return {value}
    try:
        return {str(item) for item in value if item}
    except TypeError:
        return {str(value)}


def _iter_target_devices(cfg: Dict[str, Any], entity_ids: Any) -> Iterable[Dict[str, Any]]:
    devices: list[Dict[str, Any]] = cfg.get("devices") or []
    target_ids = _normalize_entity_ids(entity_ids)

    if not target_ids:
        for device in devices:
            yield device
        return

    matched: set[str] = set()
    for device in devices:
        charger_ent = device.get(CONF_CHARGER_SWITCH)
        if charger_ent and charger_ent in target_ids:
            matched.add(charger_ent)
            yield device

    missing = target_ids - matched
    if missing:
        _LOGGER.warning(
            "Smart Charger service called with unknown entity_id(s): %s",
            ", ".join(sorted(missing)),
        )


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

async def handle_force_refresh(hass: HomeAssistant, coordinator) -> None:
    """Manuell Daten vom Coordinator abrufen."""
    _LOGGER.debug("Manual refresh requested via service.")
    refresh = getattr(coordinator, "async_throttled_refresh", None)
    if callable(refresh):
        await refresh()  # type: ignore[func-returns-value]
    else:
        await coordinator.async_request_refresh()


async def handle_start_charging(hass: HomeAssistant, cfg: Dict[str, Any], call: ServiceCall, sm: SmartChargerStateMachine) -> None:
    """Starte Ladevorgang manuell."""
    for device in _iter_target_devices(cfg, call.data.get("entity_id")):
        name = device.get("name")
        charger_ent = device.get(CONF_CHARGER_SWITCH)
        if not (name and charger_ent):
            continue

        try:
            await hass.services.async_call(
                "switch",
                "turn_on",
                {"entity_id": charger_ent},
                blocking=True,
            )
        except Exception as err:  # pragma: no cover - defensive logging
            _LOGGER.error("Failed to start manual charge for %s (%s): %s", name, charger_ent, err)
            sm.add_error(name, f"start_failed:{err}")
            continue

        sm.set_state(name, "charging")
        _LOGGER.info("Starting manual charge for %s", name)


async def handle_stop_charging(hass: HomeAssistant, cfg: Dict[str, Any], call: ServiceCall, sm: SmartChargerStateMachine) -> None:
    """Beende Ladevorgang manuell."""
    for device in _iter_target_devices(cfg, call.data.get("entity_id")):
        name = device.get("name")
        charger_ent = device.get(CONF_CHARGER_SWITCH)
        if not (name and charger_ent):
            continue

        try:
            await hass.services.async_call(
                "switch",
                "turn_off",
                {"entity_id": charger_ent},
                blocking=True,
            )
        except Exception as err:  # pragma: no cover - defensive logging
            _LOGGER.error("Failed to stop manual charge for %s (%s): %s", name, charger_ent, err)
            sm.add_error(name, f"stop_failed:{err}")
            continue

        sm.set_state(name, "idle")
        _LOGGER.info("Stopping manual charge for %s", name)


async def handle_auto_manage(
    hass: HomeAssistant,
    entry_id: str,
    cfg: Dict[str, Any],
    coordinator,
    sm: SmartChargerStateMachine,
    learning,
) -> None:
    """Automatisches Lade-Management & Lernlogik."""
    devices = cfg.get("devices") or []
    for d in devices:
        pid = d.get("name")
        if not pid:
            continue

        batt_state = _get_state(hass, d.get(CONF_BATTERY_SENSOR))
        charging_state = _get_state(hass, d.get(CONF_CHARGING_SENSOR))
        charger_ent = d.get(CONF_CHARGER_SWITCH)
        target = float(d.get(CONF_TARGET_LEVEL, 95))

        if batt_state is None:
            continue

        try:
            battery = float(batt_state)
        except ValueError:
            battery = 0.0

        # Zustandserkennung
        currently_charging = _is_charging_state(charging_state)

        # Ladebeginn erkannt
        if currently_charging and sm.states.get(pid) != "charging":
            learning.start_session(pid, battery)
            sm.set_state(pid, "charging")

        # Ladeende erkannt
        if not currently_charging and sm.states.get(pid) == "charging":
            await learning.end_session(pid, battery)
            sm.set_state(pid, "idle")

        # Autoabschaltung bei Zielwert erreicht
        if battery >= target and charger_ent:
            charger_state = _get_state(hass, charger_ent)
            if charger_state == STATE_ON:
                await hass.services.async_call("switch", "turn_off", {"entity_id": charger_ent})
                sm.set_state(pid, "completed")
                _LOGGER.info("Auto-stop: %s reached %.1f%% (target %.1f%%)", pid, battery, target)

    refresh = getattr(coordinator, "async_throttled_refresh", None)
    if callable(refresh):
        await refresh()  # type: ignore[func-returns-value]
    else:
        await coordinator.async_request_refresh()


async def handle_load_model(hass: HomeAssistant, cfg: Dict[str, Any], call: ServiceCall, learning) -> None:
    """Manuelles Laden oder Reset des Lernmodells."""
    action = call.data.get("action", "load")
    pid = call.data.get("profile_id")
    if action == "reset":
        if pid and pid in learning._data:
            learning._data[pid] = {"samples": [], "cycles": []}
            await learning.async_save()
            _LOGGER.warning("Learning data reset for %s", pid)
        elif not pid:
            learning._data.clear()
            await learning.async_save()
            _LOGGER.warning("All learning data cleared")
    else:
        await learning.async_load(pid)
        _LOGGER.info("Learning model reloaded for %s", pid or "all")

    # Automatische Bereinigung
    learning.cleanup_old_data()
