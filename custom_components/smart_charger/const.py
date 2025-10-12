"""Constants for the Smart Charger integration."""

from homeassistant.const import Platform


DOMAIN = "smart_charger"
PLATFORMS = [Platform.SENSOR]

# Update interval (seconds)
UPDATE_INTERVAL = 60

# Config keys
CONF_BATTERY_SENSOR = "battery_sensor"
CONF_CHARGER_SWITCH = "charger_switch"
CONF_CHARGING_SENSOR = "charging_sensor"
CONF_ALARM_ENTITY = "alarm_entity"
CONF_AVG_SPEED_SENSOR = "avg_speed_sensor"
CONF_PRESENCE_SENSOR = "presence_sensor"
CONF_TARGET_LEVEL = "target_level"
CONF_MIN_LEVEL = "min_level"
CONF_PRECHARGE_LEVEL = "precharge_level"
CONF_USE_PREDICTIVE_MODE = "use_predictive_mode"
CONF_NOTIFY_ENABLED = "notify_enabled"
CONF_NOTIFY_TARGETS = "notify_targets"
CONF_SUGGESTION_THRESHOLD = "suggestion_threshold"
CONF_SENSOR_STALE_SECONDS = "sensor_stale_seconds"

# New alarm configuration mode
CONF_ALARM_MODE = "alarm_mode"
ALARM_MODE_SINGLE = "single"
ALARM_MODE_PER_DAY = "per_day"

# Weekday alarm keys (used by coordinator & config flow)
CONF_ALARM_MONDAY = "alarm_entity_monday"
CONF_ALARM_TUESDAY = "alarm_entity_tuesday"
CONF_ALARM_WEDNESDAY = "alarm_entity_wednesday"
CONF_ALARM_THURSDAY = "alarm_entity_thursday"
CONF_ALARM_FRIDAY = "alarm_entity_friday"
CONF_ALARM_SATURDAY = "alarm_entity_saturday"
CONF_ALARM_SUNDAY = "alarm_entity_sunday"

# Default values
DEFAULT_SUGGESTION_THRESHOLD = 3
DEFAULT_SENSOR_STALE_SECONDS = 600
DEFAULT_TARGET_LEVEL = 95.0

# States
UNKNOWN_STATES = {"unknown", "unavailable", None}
CHARGING_STATES = {"charging", "on", "true"}
DISCHARGING_STATES = {"discharging", "off", "false"}
FULL_STATES = {"full", "complete", "done"}

# Service names
SERVICE_FORCE_REFRESH = "force_refresh"
SERVICE_START_CHARGING = "start_charging"
SERVICE_STOP_CHARGING = "stop_charging"
SERVICE_AUTO_MANAGE = "auto_manage"
SERVICE_LOAD_MODEL = "load_model"