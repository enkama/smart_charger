from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, cast

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    ALARM_MODE_PER_DAY,
    ALARM_MODE_SINGLE,
    CONF_ALARM_ENTITY,
    CONF_ALARM_MODE,
    CONF_ALARM_MONDAY,
    CONF_ALARM_FRIDAY,
    CONF_ALARM_SATURDAY,
    CONF_ALARM_SUNDAY,
    CONF_ALARM_THURSDAY,
    CONF_ALARM_TUESDAY,
    CONF_ALARM_WEDNESDAY,
    CONF_AVG_SPEED_SENSOR,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_CHARGING_SENSOR,
    CONF_MIN_LEVEL,
    CONF_NOTIFY_ENABLED,
    CONF_NOTIFY_TARGETS,
    CONF_PRESENCE_SENSOR,
    CONF_PRECHARGE_LEVEL,
    CONF_SENSOR_STALE_SECONDS,
    CONF_SUGGESTION_THRESHOLD,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    DEFAULT_SENSOR_STALE_SECONDS,
    DEFAULT_SUGGESTION_THRESHOLD,
    DEFAULT_TARGET_LEVEL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

WEEKDAY_ALARM_FIELDS: tuple[str, ...] = (
    CONF_ALARM_MONDAY,
    CONF_ALARM_TUESDAY,
    CONF_ALARM_WEDNESDAY,
    CONF_ALARM_THURSDAY,
    CONF_ALARM_FRIDAY,
    CONF_ALARM_SATURDAY,
    CONF_ALARM_SUNDAY,
)

SENSOR_SELECTOR = EntitySelector(EntitySelectorConfig(domain=["sensor"]))
CHARGING_SELECTOR = EntitySelector(
    EntitySelectorConfig(domain=["binary_sensor", "sensor"])
)
SWITCH_SELECTOR = EntitySelector(EntitySelectorConfig(domain=["switch"]))
PRESENCE_SELECTOR = EntitySelector(
    EntitySelectorConfig(domain=["person", "device_tracker"])
)
ALARM_SELECTOR = EntitySelector(
    EntitySelectorConfig(domain=["sensor", "input_datetime"])
)

OPTIONAL_ENTITY_FIELDS: tuple[str, ...] = (
    CONF_CHARGING_SENSOR,
    CONF_AVG_SPEED_SENSOR,
    CONF_PRESENCE_SENSOR,
    CONF_ALARM_ENTITY,
    *WEEKDAY_ALARM_FIELDS,
)


def _notify_selector(hass: HomeAssistant) -> SelectSelector:
    services = hass.services.async_services().get("notify", {})
    options = cast(
        list[SelectOptionDict],
        [
            {"value": service, "label": service.replace("_", " ").title()}
            for service in sorted(services)
        ],
    )
    return SelectSelector(
        SelectSelectorConfig(
            options=options,
            multiple=True,
            mode=SelectSelectorMode.DROPDOWN,
        )
    )


class SmartChargerFlowMixin:
    """Shared helpers for config and options flows."""

    hass: Any

    @staticmethod
    def _list_devices(devices: Iterable[Mapping[str, Any]]) -> str:
        names = [d.get("name") or f"Device {idx + 1}" for idx, d in enumerate(devices)]
        return "\n".join(f"• {name}" for name in names) if names else "-"

    @staticmethod
    def _duplicate_name(
        devices: Sequence[Mapping[str, Any]],
        name: str,
        *,
        skip_idx: Optional[int] = None,
    ) -> bool:
        for idx, device in enumerate(devices):
            if skip_idx is not None and idx == skip_idx:
                continue
            if device.get("name") == name:
                return True
        return False

    def _level_errors(self, data: Mapping[str, Any]) -> dict[str, str]:
        errors: dict[str, str] = {}
        try:
            target = float(data.get(CONF_TARGET_LEVEL, DEFAULT_TARGET_LEVEL))
            min_level = float(data.get(CONF_MIN_LEVEL, 30))
            precharge = float(data.get(CONF_PRECHARGE_LEVEL, 50))
        except (TypeError, ValueError):
            errors[CONF_TARGET_LEVEL] = "invalid_level"
            return errors

        if not (1 <= target <= 100):
            errors[CONF_TARGET_LEVEL] = "invalid_level"
        if not (0 <= min_level < target):
            errors[CONF_MIN_LEVEL] = "invalid_level"
        if not (min_level <= precharge <= target):
            errors[CONF_PRECHARGE_LEVEL] = "invalid_level"
        return errors

    @staticmethod
    def _alarm_errors(data: Mapping[str, Any]) -> dict[str, str]:
        errors: dict[str, str] = {}
        mode = data.get(CONF_ALARM_MODE, ALARM_MODE_SINGLE)
        if mode == ALARM_MODE_SINGLE:
            if not data.get(CONF_ALARM_ENTITY):
                errors[CONF_ALARM_MODE] = "alarm_required_single"
        elif mode == ALARM_MODE_PER_DAY:
            if not any(data.get(field) for field in WEEKDAY_ALARM_FIELDS):
                errors[CONF_ALARM_MODE] = "alarm_required_per_day"
        return errors

    @staticmethod
    def _optional_with_default(key: str, data: Mapping[str, Any], default: Any) -> Any:
        return vol.Optional(key, default=data.get(key, default))

    @staticmethod
    def _required_with_default(key: str, data: Optional[Mapping[str, Any]]) -> Any:
        if data and key in data:
            return vol.Required(key, default=data[key])
        return vol.Required(key)

    @staticmethod
    def _optional_selector(key: str, defaults: Optional[Mapping[str, Any]]) -> Any:
        if defaults and defaults.get(key):
            return vol.Optional(key, default=defaults[key])
        return vol.Optional(key)

    @staticmethod
    def _sanitize_optional_entities(data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(data)
        for key in OPTIONAL_ENTITY_FIELDS:
            if not cleaned.get(key):
                cleaned.pop(key, None)
        return cleaned

    def _build_basic_schema(
        self, device: Optional[Mapping[str, Any]] = None, *, include_name: bool = True
    ) -> vol.Schema:
        fields: dict[Any, Any] = {}
        if include_name:
            key = self._required_with_default("name", device)
            fields[key] = str
        key = self._required_with_default(CONF_BATTERY_SENSOR, device)
        fields[key] = SENSOR_SELECTOR
        key = self._required_with_default(CONF_CHARGER_SWITCH, device)
        fields[key] = SWITCH_SELECTOR
        key = self._optional_selector(CONF_CHARGING_SENSOR, device)
        fields[key] = CHARGING_SELECTOR
        key = self._optional_selector(CONF_PRESENCE_SENSOR, device)
        fields[key] = PRESENCE_SELECTOR
        return vol.Schema(fields)

    def _build_target_schema(
        self, device: Optional[Mapping[str, Any]] = None
    ) -> vol.Schema:
        defaults = device or {}
        return vol.Schema(
            {
                self._optional_with_default(
                    CONF_TARGET_LEVEL, defaults, DEFAULT_TARGET_LEVEL
                ): vol.Coerce(float),
                self._optional_with_default(CONF_MIN_LEVEL, defaults, 30): vol.Coerce(
                    float
                ),
                self._optional_with_default(
                    CONF_PRECHARGE_LEVEL, defaults, 50
                ): vol.Coerce(float),
                self._optional_selector(
                    CONF_AVG_SPEED_SENSOR, defaults
                ): SENSOR_SELECTOR,
                vol.Optional(
                    CONF_USE_PREDICTIVE_MODE,
                    default=defaults.get(CONF_USE_PREDICTIVE_MODE, True),
                ): bool,
            }
        )

    def _build_alarm_schema(
        self, device: Optional[Mapping[str, Any]] = None
    ) -> vol.Schema:
        defaults = device or {}
        notify_selector = _notify_selector(self.hass)
        fields: dict[Any, Any] = {
            vol.Required(
                CONF_ALARM_MODE,
                default=defaults.get(CONF_ALARM_MODE, ALARM_MODE_SINGLE),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[ALARM_MODE_SINGLE, ALARM_MODE_PER_DAY],
                    translation_key="alarm_mode",
                    multiple=False,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            self._optional_selector(CONF_ALARM_ENTITY, defaults): ALARM_SELECTOR,
        }
        for field in WEEKDAY_ALARM_FIELDS:
            fields[self._optional_selector(field, defaults)] = ALARM_SELECTOR
        fields[self._optional_with_default(CONF_NOTIFY_ENABLED, defaults, False)] = bool
        fields[self._optional_with_default(CONF_NOTIFY_TARGETS, defaults, [])] = (
            notify_selector
        )
        fields[
            self._optional_with_default(
                CONF_SUGGESTION_THRESHOLD, defaults, DEFAULT_SUGGESTION_THRESHOLD
            )
        ] = NumberSelector(NumberSelectorConfig(min=1, max=10, step=1))
        fields[
            self._optional_with_default(
                CONF_SENSOR_STALE_SECONDS, defaults, DEFAULT_SENSOR_STALE_SECONDS
            )
        ] = NumberSelector(
            NumberSelectorConfig(min=60, max=3600, step=60, unit_of_measurement="s")
        )
        return vol.Schema(fields)

    def _build_full_schema(self, device: Mapping[str, Any]) -> vol.Schema:
        schema = {}
        schema.update(self._build_basic_schema(device).schema)
        schema.update(self._build_target_schema(device).schema)
        schema.update(self._build_alarm_schema(device).schema)
        return vol.Schema(schema)

    async def _async_remove_from_registry(self, name: str) -> None:
        try:
            registry = dr.async_get(self.hass)
            identifier = (DOMAIN, name.lower().replace(" ", "_"))
            if device_entry := registry.async_get_device({identifier}):
                registry.async_remove_device(device_entry.id)
                _LOGGER.debug("Removed device '%s' from registry", name)
        except Exception as err:
            _LOGGER.warning("Could not remove device '%s' from registry: %s", name, err)


class SmartChargerConfigFlow(
    SmartChargerFlowMixin, config_entries.ConfigFlow, domain=DOMAIN
):
    """Handle Smart Charger config flow."""

    VERSION = 1

    def __init__(self) -> None:
        self._devices: list[Dict[str, Any]] = []
        self._new_device: Dict[str, Any] = {}

    def _get_devices(self) -> list[Dict[str, Any]]:
        if self._devices:
            return self._devices
        domain_store = self.hass.data.setdefault(DOMAIN, {})
        cached = domain_store.get("flow_devices")
        if cached:
            self._devices = deepcopy(cached)
        return self._devices

    def _save_devices(self, devices: list[Dict[str, Any]]) -> None:
        self._devices = devices
        self.hass.data.setdefault(DOMAIN, {})["flow_devices"] = deepcopy(devices)

    async def async_step_user(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        menu = (
            ["add_device"]
            if not devices
            else ["add_device", "edit_device", "delete_device"]
        )
        return self.async_show_menu(
            step_id="user",
            menu_options=menu,
            description_placeholders={"devices": self._list_devices(devices)},
        )

    async def async_step_add_device(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        schema = self._build_basic_schema(include_name=True)
        errors: dict[str, str] = {}

        if user_input:
            name = str(user_input.get("name", "")).strip()
            if not name:
                errors["name"] = "name_empty"
            elif self._duplicate_name(devices, name):
                errors["name"] = "name_exists"

            if not errors:
                self._new_device = self._sanitize_optional_entities(dict(user_input))
                return await self.async_step_add_device_page_2()

        return self.async_show_form(
            step_id="add_device", data_schema=schema, errors=errors
        )

    async def async_step_add_device_page_2(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        schema = self._build_target_schema(self._new_device)
        errors: dict[str, str] = {}

        if user_input:
            errors = self._level_errors(user_input)
            if not errors:
                self._new_device.update(user_input)
                self._new_device = self._sanitize_optional_entities(self._new_device)
                return await self.async_step_add_device_page_3()

        return self.async_show_form(
            step_id="add_device_page_2", data_schema=schema, errors=errors
        )

    async def async_step_add_device_page_3(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        schema = self._build_alarm_schema(self._new_device)
        errors: dict[str, str] = {}

        if user_input:
            errors = self._alarm_errors(user_input)
            if not errors:
                device = self._sanitize_optional_entities(
                    {**self._new_device, **user_input}
                )
                devices = self._get_devices()
                devices.append(device)
                self._save_devices(devices)

                if not self._async_current_entries():
                    return await self.async_step_finish()
                return await self.async_step_user()

        return self.async_show_form(
            step_id="add_device_page_3",
            data_schema=schema,
            errors=errors,
            description_placeholders={"info": "Configure alarms and notifications"},
        )

    async def async_step_edit_device(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        if not devices:
            return await self.async_step_user()

        schema = vol.Schema(
            {vol.Required("idx"): vol.In({i: d["name"] for i, d in enumerate(devices)})}
        )
        if user_input is not None:
            return await self.async_step_edit_device_details(user_input)

        return self.async_show_form(step_id="edit_device", data_schema=schema)

    async def async_step_edit_device_details(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        if not user_input or "idx" not in user_input:
            return await self.async_step_user()

        idx = user_input["idx"]
        if not isinstance(idx, int) or idx >= len(devices):
            return await self.async_step_user()

        device = devices[idx]
        schema = self._build_full_schema(device)
        errors: dict[str, str] = {}

        if len(user_input) > 1:
            name = str(user_input.get("name", "")).strip()
            if not name:
                errors["name"] = "name_empty"
            elif self._duplicate_name(devices, name, skip_idx=idx):
                errors["name"] = "name_exists"

            errors.update(self._level_errors(user_input))
            errors.update(self._alarm_errors(user_input))

            if not errors:
                updated = self._sanitize_optional_entities(dict(user_input))
                devices[idx] = updated
                self._save_devices(devices)
                return await self.async_step_user()

        return self.async_show_form(
            step_id="edit_device_details",
            data_schema=schema,
            errors=errors,
            description_placeholders={"device_name": device.get("name", "")},
        )

    async def async_step_delete_device(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        if not devices:
            return await self.async_step_user()

        schema = vol.Schema(
            {vol.Required("idx"): vol.In({i: d["name"] for i, d in enumerate(devices)})}
        )
        if user_input and "idx" in user_input:
            idx = int(user_input["idx"])
            if 0 <= idx < len(devices):
                self._idx_to_delete = idx
                return await self.async_step_confirm_delete()

        return self.async_show_form(step_id="delete_device", data_schema=schema)

    async def async_step_confirm_delete(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        idx = getattr(self, "_idx_to_delete", None)
        if idx is None or idx >= len(devices):
            return await self.async_step_user()

        schema = vol.Schema({vol.Required("confirm"): bool})
        if user_input and user_input.get("confirm"):
            dev_name = devices[idx]["name"]
            devices.pop(idx)
            self._save_devices(devices)
            await self._async_remove_from_registry(dev_name)
            return await self.async_step_user()

        return self.async_show_form(
            step_id="confirm_delete",
            data_schema=schema,
            description_placeholders={"device_name": devices[idx]["name"]},
        )

    async def async_step_finish(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        devices = self._get_devices()
        return self.async_create_entry(
            title="Smart Charger",
            data={"devices": devices},
            options={"initialized": True},
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        return SmartChargerOptionsFlowHandler(config_entry)


class SmartChargerOptionsFlowHandler(SmartChargerFlowMixin, config_entries.OptionsFlow):
    """Handle Smart Charger options flow."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry
        self.devices: list[Dict[str, Any]] = deepcopy(
            config_entry.data.get("devices", [])
        )

    async def async_step_init(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        if user_input:
            action = user_input.get("action")
            if action == "edit_device":
                return await self.async_step_edit_device()
            if action == "delete_device":
                return await self.async_step_delete_device()
            return self.async_create_entry(title="", data={})

        info = f"{len(self.devices)} Smart Charger device(s) configured."
        return self.async_show_menu(
            step_id="init",
            menu_options=["edit_device", "delete_device"],
            description_placeholders={"info": info},
        )

    async def async_step_edit_device(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        if not self.devices:
            return self.async_abort(reason="no_devices")

        schema = vol.Schema(
            {
                vol.Required("idx"): vol.In(
                    {i: d["name"] for i, d in enumerate(self.devices)}
                )
            }
        )
        if user_input:
            return await self.async_step_edit_device_details(user_input)
        return self.async_show_form(step_id="edit_device", data_schema=schema)

    async def async_step_edit_device_details(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        if not user_input or "idx" not in user_input:
            return self.async_abort(reason="invalid_device")

        idx = user_input["idx"]
        if not isinstance(idx, int) or idx >= len(self.devices):
            return self.async_abort(reason="invalid_device")

        device = self.devices[idx]
        schema = self._build_full_schema(device)
        errors: dict[str, str] = {}

        if len(user_input) > 1:
            name = str(user_input.get("name", "")).strip()
            if not name:
                errors["name"] = "name_empty"
            elif self._duplicate_name(self.devices, name, skip_idx=idx):
                errors["name"] = "name_exists"

            errors.update(self._level_errors(user_input))
            errors.update(self._alarm_errors(user_input))

            if not errors:
                self.devices[idx] = self._sanitize_optional_entities(dict(user_input))
                self.hass.config_entries.async_update_entry(
                    self.config_entry, data={"devices": deepcopy(self.devices)}
                )
                return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="edit_device_details",
            data_schema=schema,
            errors=errors,
            description_placeholders={"device_name": device.get("name", "")},
        )

    async def async_step_delete_device(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        if not self.devices:
            return self.async_abort(reason="no_devices")

        schema = vol.Schema(
            {
                vol.Required("idx"): vol.In(
                    {i: d["name"] for i, d in enumerate(self.devices)}
                )
            }
        )
        if user_input and "idx" in user_input:
            idx = int(user_input["idx"])
            if 0 <= idx < len(self.devices):
                name = self.devices[idx]["name"]
                self.devices.pop(idx)
                self.hass.config_entries.async_update_entry(
                    self.config_entry, data={"devices": deepcopy(self.devices)}
                )
                await self._async_remove_from_registry(name)
                return self.async_create_entry(title="", data={})

        return self.async_show_form(step_id="delete_device", data_schema=schema)
