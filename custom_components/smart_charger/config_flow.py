from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, cast

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
    CONF_ALARM_FRIDAY,
    CONF_ALARM_MODE,
    CONF_ALARM_MONDAY,
    CONF_ALARM_SATURDAY,
    CONF_ALARM_SUNDAY,
    CONF_ALARM_THURSDAY,
    CONF_ALARM_TUESDAY,
    CONF_ALARM_WEDNESDAY,
    CONF_AVG_SPEED_SENSOR,
    CONF_BATTERY_SENSOR,
    CONF_CHARGER_SWITCH,
    CONF_CHARGING_SENSOR,
    CONF_LEARNING_RECENT_SAMPLE_HOURS,
    CONF_MIN_LEVEL,
    CONF_NOTIFY_ENABLED,
    CONF_NOTIFY_TARGETS,
    CONF_PRECHARGE_COUNTDOWN_WINDOW,
    CONF_PRECHARGE_LEVEL,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_PRECHARGE_MARGIN_ON,
    CONF_PRESENCE_SENSOR,
    CONF_SENSOR_STALE_SECONDS,
    CONF_SMART_START_MARGIN,
    CONF_SUGGESTION_THRESHOLD,
    CONF_SWITCH_CONFIRMATION_COUNT,
    CONF_SWITCH_THROTTLE_SECONDS,
    CONF_TARGET_LEVEL,
    CONF_USE_PREDICTIVE_MODE,
    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    DEFAULT_PRECHARGE_COUNTDOWN_WINDOW,
    DEFAULT_PRECHARGE_MARGIN_OFF,
    DEFAULT_PRECHARGE_MARGIN_ON,
    DEFAULT_SENSOR_STALE_SECONDS,
    DEFAULT_SMART_START_MARGIN,
    DEFAULT_SUGGESTION_THRESHOLD,
    DEFAULT_SWITCH_CONFIRMATION_COUNT,
    DEFAULT_SWITCH_THROTTLE_SECONDS,
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
ALARM_MODE_SELECTOR = SelectSelector(
    SelectSelectorConfig(
        options=[ALARM_MODE_SINGLE, ALARM_MODE_PER_DAY],
        translation_key="alarm_mode",
        multiple=False,
        mode=SelectSelectorMode.DROPDOWN,
    )
)

OPTIONAL_ENTITY_FIELDS: tuple[str, ...] = (
    CONF_CHARGING_SENSOR,
    CONF_AVG_SPEED_SENSOR,
    CONF_PRESENCE_SENSOR,
    CONF_ALARM_ENTITY,
    *WEEKDAY_ALARM_FIELDS,
)


MISSING = object()


@dataclass(frozen=True)
class SchemaField:
    """Definition for a single schema field."""

    key: str
    required: bool = False
    validator: Any | None = None
    selector: Any | None = None
    selector_factory: Callable[["SmartChargerFlowMixin"], Any] | None = None
    default: Any = MISSING
    default_factory: Callable[[], Any] | None = None
    existing_only: bool = False

    def build(
        self,
        flow: "SmartChargerFlowMixin",
        defaults: Optional[Mapping[str, Any]],
    ) -> tuple[Any, Any]:
        value = MISSING
        if defaults and self.key in defaults:
            candidate = defaults[self.key]
            if self.existing_only and not candidate:
                value = MISSING
            else:
                value = candidate
        elif not self.existing_only:
            if self.default_factory is not None:
                value = self.default_factory()
            elif self.default is not MISSING:
                value = self.default

        field_cls = vol.Required if self.required else vol.Optional
        field = (
            field_cls(self.key, default=value)
            if value is not MISSING
            else field_cls(self.key)
        )

        validator = self.selector or self.validator
        if self.selector_factory is not None:
            validator = self.selector_factory(flow)
        if validator is None:
            raise ValueError(f"No validator configured for field '{self.key}'")
        return field, validator


def _notify_selector_from_flow(flow: "SmartChargerFlowMixin") -> SelectSelector:
    return _notify_selector(flow.hass)


NAME_FIELD = SchemaField("name", required=True, validator=str)
BASIC_DEVICE_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(CONF_BATTERY_SENSOR, required=True, selector=SENSOR_SELECTOR),
    SchemaField(CONF_CHARGER_SWITCH, required=True, selector=SWITCH_SELECTOR),
    SchemaField(CONF_CHARGING_SENSOR, selector=CHARGING_SELECTOR, existing_only=True),
    SchemaField(CONF_PRESENCE_SENSOR, selector=PRESENCE_SELECTOR, existing_only=True),
)
BASIC_TARGET_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(
        CONF_TARGET_LEVEL,
        validator=vol.Coerce(float),
        default=DEFAULT_TARGET_LEVEL,
    ),
    SchemaField(CONF_MIN_LEVEL, validator=vol.Coerce(float), default=30),
    SchemaField(
        CONF_PRECHARGE_LEVEL,
        validator=vol.Coerce(float),
        default=50,
    ),
    SchemaField(CONF_AVG_SPEED_SENSOR, selector=SENSOR_SELECTOR, existing_only=True),
    SchemaField(CONF_USE_PREDICTIVE_MODE, validator=bool, default=True),
)
ADVANCED_DEVICE_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(
        CONF_PRECHARGE_MARGIN_ON,
        selector=NumberSelector(
            NumberSelectorConfig(min=0, max=10, step=0.1, unit_of_measurement="%")
        ),
        default=DEFAULT_PRECHARGE_MARGIN_ON,
    ),
    SchemaField(
        CONF_PRECHARGE_MARGIN_OFF,
        selector=NumberSelector(
            NumberSelectorConfig(min=0, max=10, step=0.1, unit_of_measurement="%")
        ),
        default=DEFAULT_PRECHARGE_MARGIN_OFF,
    ),
    SchemaField(
        CONF_SMART_START_MARGIN,
        selector=NumberSelector(
            NumberSelectorConfig(min=0, max=10, step=0.1, unit_of_measurement="%")
        ),
        default=DEFAULT_SMART_START_MARGIN,
    ),
    SchemaField(
        CONF_PRECHARGE_COUNTDOWN_WINDOW,
        selector=NumberSelector(
            NumberSelectorConfig(min=0, max=20, step=0.5, unit_of_measurement="%")
        ),
        default=DEFAULT_PRECHARGE_COUNTDOWN_WINDOW,
    ),
    SchemaField(
        CONF_SUGGESTION_THRESHOLD,
        selector=NumberSelector(NumberSelectorConfig(min=1, max=10, step=1)),
        default=DEFAULT_SUGGESTION_THRESHOLD,
    ),
    SchemaField(
        CONF_SENSOR_STALE_SECONDS,
        selector=NumberSelector(
            NumberSelectorConfig(min=60, max=3600, step=60, unit_of_measurement="s")
        ),
        default=DEFAULT_SENSOR_STALE_SECONDS,
    ),
    SchemaField(
        CONF_LEARNING_RECENT_SAMPLE_HOURS,
        selector=NumberSelector(
            NumberSelectorConfig(
                min=0.5,
                max=48,
                step=0.5,
                unit_of_measurement="h",
            )
        ),
        default=DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
    ),
    SchemaField(
        CONF_SWITCH_THROTTLE_SECONDS,
        selector=NumberSelector(
            NumberSelectorConfig(min=1, max=600, step=1, unit_of_measurement="s")
        ),
        default=DEFAULT_SWITCH_THROTTLE_SECONDS,
    ),
    SchemaField(
        CONF_SWITCH_CONFIRMATION_COUNT,
        selector=NumberSelector(NumberSelectorConfig(min=1, max=5, step=1)),
        default=DEFAULT_SWITCH_CONFIRMATION_COUNT,
    ),
)
ALARM_FIELDS: tuple[SchemaField, ...] = (
    SchemaField(
        CONF_ALARM_MODE,
        required=True,
        selector=ALARM_MODE_SELECTOR,
        default=ALARM_MODE_SINGLE,
    ),
    SchemaField(CONF_ALARM_ENTITY, selector=ALARM_SELECTOR, existing_only=True),
    *(
        SchemaField(field, selector=ALARM_SELECTOR, existing_only=True)
        for field in WEEKDAY_ALARM_FIELDS
    ),
    SchemaField(CONF_NOTIFY_ENABLED, validator=bool, default=False),
    SchemaField(
        CONF_NOTIFY_TARGETS,
        selector_factory=_notify_selector_from_flow,
        default_factory=list,
    ),
)

ADVANCED_FIELD_KEYS = {
    CONF_PRECHARGE_MARGIN_ON,
    CONF_PRECHARGE_MARGIN_OFF,
    CONF_SMART_START_MARGIN,
    CONF_PRECHARGE_COUNTDOWN_WINDOW,
    CONF_SUGGESTION_THRESHOLD,
    CONF_SENSOR_STALE_SECONDS,
    CONF_LEARNING_RECENT_SAMPLE_HOURS,
}


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
    _advanced_idx: Optional[int]

    def _schema_from_fields(
        self,
        fields: Iterable[SchemaField],
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> vol.Schema:
        schema_fields: dict[Any, Any] = {}
        for field in fields:
            field_key, validator = field.build(self, defaults)
            schema_fields[field_key] = validator
        return vol.Schema(schema_fields)

    @staticmethod
    def _list_devices(devices: Iterable[Mapping[str, Any]]) -> str:
        names = [d.get("name") or f"Device {idx + 1}" for idx, d in enumerate(devices)]
        return "\n".join(f"• {name}" for name in names) if names else "-"

    def _device_count_message(self, count: int) -> str:
        language = getattr(getattr(self.hass, "config", None), "language", None) or "en"
        primary = language.split("-")[0].lower()

        if primary == "de":
            if count == 0:
                return "Keine Smart-Charger-Ger\u00e4te konfiguriert."
            if count == 1:
                return "1 Smart-Charger-Ger\u00e4t konfiguriert."
            return f"{count} Smart-Charger-Ger\u00e4te konfiguriert."

        if count == 0:
            return "No Smart Charger devices configured."
        if count == 1:
            return "1 Smart Charger device configured."
        return f"{count} Smart Charger devices configured."

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
            precharge_margin_on = float(
                data.get(CONF_PRECHARGE_MARGIN_ON, DEFAULT_PRECHARGE_MARGIN_ON)
            )
            precharge_margin_off = float(
                data.get(CONF_PRECHARGE_MARGIN_OFF, DEFAULT_PRECHARGE_MARGIN_OFF)
            )
            smart_start_margin = float(
                data.get(CONF_SMART_START_MARGIN, DEFAULT_SMART_START_MARGIN)
            )
            countdown_window = float(
                data.get(
                    CONF_PRECHARGE_COUNTDOWN_WINDOW,
                    DEFAULT_PRECHARGE_COUNTDOWN_WINDOW,
                )
            )
            learning_window = float(
                data.get(
                    CONF_LEARNING_RECENT_SAMPLE_HOURS,
                    DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
                )
            )
        except (TypeError, ValueError):
            errors[CONF_TARGET_LEVEL] = "invalid_level"
            return errors

        if not (1 <= target <= 100):
            errors[CONF_TARGET_LEVEL] = "invalid_level"
        if not (0 <= min_level < target):
            errors[CONF_MIN_LEVEL] = "invalid_level"
        if not (min_level <= precharge <= target):
            errors[CONF_PRECHARGE_LEVEL] = "invalid_level"
        if precharge_margin_on < 0:
            errors[CONF_PRECHARGE_MARGIN_ON] = "invalid_margin"
        if precharge_margin_off < precharge_margin_on:
            errors[CONF_PRECHARGE_MARGIN_OFF] = "invalid_margin"
        if precharge_margin_off < 0:
            errors[CONF_PRECHARGE_MARGIN_OFF] = "invalid_margin"
        if smart_start_margin < 0:
            errors[CONF_SMART_START_MARGIN] = "invalid_margin"
        if countdown_window < 0:
            errors[CONF_PRECHARGE_COUNTDOWN_WINDOW] = "invalid_margin"
        if learning_window < 0.25 or learning_window > 48:
            errors[CONF_LEARNING_RECENT_SAMPLE_HOURS] = "invalid_margin"
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
    def _sanitize_optional_entities(data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(data)
        for key in OPTIONAL_ENTITY_FIELDS:
            if not cleaned.get(key):
                cleaned.pop(key, None)
        return cleaned

    def _build_basic_schema(
        self, device: Optional[Mapping[str, Any]] = None, *, include_name: bool = True
    ) -> vol.Schema:
        fields: tuple[SchemaField, ...]
        fields = BASIC_DEVICE_FIELDS
        if include_name:
            fields = (NAME_FIELD, *fields)
        return self._schema_from_fields(fields, device)

    def _build_target_schema(
        self,
        device: Optional[Mapping[str, Any]] = None,
        *,
        include_advanced: bool = False,
    ) -> vol.Schema:
        fields: tuple[SchemaField, ...]
        if include_advanced:
            fields = (*BASIC_TARGET_FIELDS, *ADVANCED_DEVICE_FIELDS)
        else:
            fields = BASIC_TARGET_FIELDS
        return self._schema_from_fields(fields, device)

    def _build_alarm_schema(
        self, device: Optional[Mapping[str, Any]] = None
    ) -> vol.Schema:
        return self._schema_from_fields(ALARM_FIELDS, device)

    def _build_full_schema(self, device: Mapping[str, Any]) -> vol.Schema:
        fields = (
            NAME_FIELD,
            *BASIC_DEVICE_FIELDS,
            *BASIC_TARGET_FIELDS,
            *ALARM_FIELDS,
        )
        return self._schema_from_fields(fields, device)

    def _build_basic_options_schema(self, device: Mapping[str, Any]) -> vol.Schema:
        fields = (
            NAME_FIELD,
            *BASIC_DEVICE_FIELDS,
            *BASIC_TARGET_FIELDS,
            *ALARM_FIELDS,
        )
        return self._schema_from_fields(fields, device)

    def _build_device_advanced_schema(
        self,
        device: Mapping[str, Any],
        *,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> vol.Schema:
        merged: Dict[str, Any] = {}
        if defaults:
            merged.update(defaults)
        merged.update(device)
        return self._schema_from_fields(ADVANCED_DEVICE_FIELDS, merged)

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

            merged = {**device, **user_input}
            errors.update(self._level_errors(merged))
            errors.update(self._alarm_errors(merged))

            if not errors:
                updated = self._sanitize_optional_entities({**device, **user_input})
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
            if action == "advanced_settings":
                return await self.async_step_advanced_settings()
            return self.async_create_entry(title="", data={})

        info = self._device_count_message(len(self.devices))
        return self.async_show_menu(
            step_id="init",
            menu_options=["edit_device", "advanced_settings", "delete_device"],
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
        schema = self._build_basic_options_schema(device)
        errors: dict[str, str] = {}

        if len(user_input) > 1:
            name = str(user_input.get("name", "")).strip()
            if not name:
                errors["name"] = "name_empty"
            elif self._duplicate_name(self.devices, name, skip_idx=idx):
                errors["name"] = "name_exists"

            merged = {**device, **user_input}
            errors.update(self._level_errors(merged))
            errors.update(self._alarm_errors(merged))

            if not errors:
                updated = self._sanitize_optional_entities({**device, **user_input})
                self.devices[idx] = updated
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

    async def async_step_advanced_settings(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        if user_input and user_input.get("next_step_id"):
            next_step = user_input["next_step_id"]
            handler = getattr(self, f"async_step_{next_step}", None)
            if handler is not None:
                return await handler()

        info = self._device_count_message(len(self.devices))
        return self.async_show_menu(
            step_id="advanced_settings",
            menu_options=[
                "advanced_settings_device",
            ],
            description_placeholders={"info": info},
        )

    async def async_step_advanced_settings_device(
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
                self._advanced_idx = idx
                return await self.async_step_advanced_settings_device_details()

        return self.async_show_form(
            step_id="advanced_settings_device",
            data_schema=schema,
        )

    async def async_step_advanced_settings_device_details(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> config_entries.ConfigFlowResult:
        idx = getattr(self, "_advanced_idx", None)
        if idx is None or idx >= len(self.devices):
            return self.async_abort(reason="invalid_device")

        device = self.devices[idx]
        defaults = {
            CONF_LEARNING_RECENT_SAMPLE_HOURS: self.config_entry.options.get(
                CONF_LEARNING_RECENT_SAMPLE_HOURS,
                DEFAULT_LEARNING_RECENT_SAMPLE_HOURS,
            )
        }
        schema = self._build_device_advanced_schema(device, defaults=defaults)
        errors: dict[str, str] = {}

        if user_input:
            merged = {**device, **user_input}
            errors = {
                key: value
                for key, value in self._level_errors(merged).items()
                if key in ADVANCED_FIELD_KEYS
            }

            if not errors:
                updated = self._sanitize_optional_entities(merged)
                self.devices[idx] = updated
                options_kwargs: dict[str, Any] = {}
                if CONF_LEARNING_RECENT_SAMPLE_HOURS in self.config_entry.options:
                    new_options = dict(self.config_entry.options)
                    new_options.pop(CONF_LEARNING_RECENT_SAMPLE_HOURS, None)
                    options_kwargs["options"] = new_options
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={"devices": deepcopy(self.devices)},
                    **options_kwargs,
                )
                self._advanced_idx = None
                return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="advanced_settings_device_details",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "device_name": device.get("name", ""),
            },
        )
