"""Constants and defaults for the Smart Charger integration."""

from homeassistant.const import Platform

DOMAIN = "smart_charger"
PLATFORMS = [Platform.SENSOR]

# Update interval expressed in seconds.
UPDATE_INTERVAL = 60

# Configuration keys shared across the integration.
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
CONF_PRECHARGE_MARGIN_ON = "precharge_margin_on"
CONF_PRECHARGE_MARGIN_OFF = "precharge_margin_off"
CONF_SMART_START_MARGIN = "smart_start_margin"
CONF_PRECHARGE_COUNTDOWN_WINDOW = "precharge_countdown_window"
# Minimum percent drop required after a precharge release before re-activating
CONF_PRECHARGE_MIN_DROP_PERCENT = "precharge_min_drop_percent"
# Default changed per user request: 10% required drop before re-activation
DEFAULT_PRECHARGE_MIN_DROP_PERCENT = 10.0
# Cooldown (minutes) after a precharge release before re-activating
# Keep the option key unchanged for backwards compatibility (stored as minutes)
CONF_PRECHARGE_COOLDOWN_MINUTES = "precharge_cooldown_minutes"
# Default cooldown (minutes) per user request
DEFAULT_PRECHARGE_COOLDOWN_MINUTES = 60.0
CONF_LEARNING_RECENT_SAMPLE_HOURS = "learning_recent_sample_hours"
CONF_SWITCH_THROTTLE_SECONDS = "switch_throttle_seconds"
CONF_NOTIFY_ENABLED = "notify_enabled"
CONF_NOTIFY_TARGETS = "notify_targets"
CONF_SUGGESTION_THRESHOLD = "suggestion_threshold"
CONF_SENSOR_STALE_SECONDS = "sensor_stale_seconds"

# Alarm configuration modes and options.
CONF_ALARM_MODE = "alarm_mode"
ALARM_MODE_SINGLE = "single"
ALARM_MODE_PER_DAY = "per_day"

# Weekday-specific alarm entity keys used by the coordinator and config flow.
CONF_ALARM_MONDAY = "alarm_entity_monday"
CONF_ALARM_TUESDAY = "alarm_entity_tuesday"
CONF_ALARM_WEDNESDAY = "alarm_entity_wednesday"
CONF_ALARM_THURSDAY = "alarm_entity_thursday"
CONF_ALARM_FRIDAY = "alarm_entity_friday"
CONF_ALARM_SATURDAY = "alarm_entity_saturday"
CONF_ALARM_SUNDAY = "alarm_entity_sunday"

# Default thresholds used when configuration overrides are absent.
DEFAULT_SUGGESTION_THRESHOLD = 3
DEFAULT_SENSOR_STALE_SECONDS = 600
DEFAULT_TARGET_LEVEL = 95.0
DEFAULT_PRECHARGE_MARGIN_ON = 1.0
DEFAULT_PRECHARGE_MARGIN_OFF = 2.0
DEFAULT_SMART_START_MARGIN = 2.0
DEFAULT_PRECHARGE_COUNTDOWN_WINDOW = 5.0
# Minimum percent drop required after a precharge release before re-activating
CONF_PRECHARGE_MIN_DROP_PERCENT = "precharge_min_drop_percent"
DEFAULT_PRECHARGE_MIN_DROP_PERCENT = 10.0
DEFAULT_LEARNING_RECENT_SAMPLE_HOURS = 4.0
DEFAULT_SWITCH_THROTTLE_SECONDS = 120.0
CONF_SWITCH_CONFIRMATION_COUNT = "switch_confirmation_count"
# Default confirmation count required before issuing a switch call. Set to 1
# for backwards compatibility so existing behavior (immediate actions) is
# preserved unless users explicitly configure a higher value.
DEFAULT_SWITCH_CONFIRMATION_COUNT = 1

# Adaptive throttle tuning defaults
CONF_ADAPTIVE_THROTTLE_ENABLED = "adaptive_throttle_enabled"
CONF_ADAPTIVE_THROTTLE_MULTIPLIER = "adaptive_throttle_multiplier"
CONF_ADAPTIVE_THROTTLE_MIN_SECONDS = "adaptive_throttle_min_seconds"
CONF_ADAPTIVE_THROTTLE_DURATION_SECONDS = "adaptive_throttle_duration_seconds"
CONF_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS = "adaptive_flipflop_window_seconds"
CONF_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD = "adaptive_flipflop_warn_threshold"

DEFAULT_ADAPTIVE_THROTTLE_ENABLED = True
DEFAULT_ADAPTIVE_THROTTLE_MULTIPLIER = 2.0
DEFAULT_ADAPTIVE_THROTTLE_MIN_SECONDS = 120.0
DEFAULT_ADAPTIVE_THROTTLE_DURATION_SECONDS = 600.0
DEFAULT_ADAPTIVE_FLIPFLOP_WINDOW_SECONDS = 300.0
DEFAULT_ADAPTIVE_FLIPFLOP_WARN_THRESHOLD = 3

# Backoff / variable multiplier tuning
CONF_ADAPTIVE_THROTTLE_BACKOFF_STEP = "adaptive_throttle_backoff_step"
CONF_ADAPTIVE_THROTTLE_MAX_MULTIPLIER = "adaptive_throttle_max_multiplier"

DEFAULT_ADAPTIVE_THROTTLE_BACKOFF_STEP = 0.5
DEFAULT_ADAPTIVE_THROTTLE_MAX_MULTIPLIER = 5.0

# Adaptive mode presets to tune conservativeness/aggressiveness
CONF_ADAPTIVE_THROTTLE_MODE = "adaptive_throttle_mode"
ADAPTIVE_MODE_CONSERVATIVE = "conservative"
ADAPTIVE_MODE_NORMAL = "normal"
ADAPTIVE_MODE_AGGRESSIVE = "aggressive"
DEFAULT_ADAPTIVE_THROTTLE_MODE = ADAPTIVE_MODE_NORMAL

# EWMA smoothing factor used to compute rolling average of flip-flop rate
CONF_ADAPTIVE_EWMA_ALPHA = "adaptive_ewma_alpha"
DEFAULT_ADAPTIVE_EWMA_ALPHA = 0.3

# Learning / prediction defaults.
LEARNING_CACHE_TTL = 60  # seconds
LEARNING_MIN_SPEED = 0.1
LEARNING_MAX_SPEED = 80.0
LEARNING_DEFAULT_SPEED = 1.0
LEARNING_EMA_ALPHA = 0.6
# Drain-rate sanity guard to avoid runaway predictions when sensors glitch.
MAX_OBSERVED_DRAIN_RATE = 6.0

# Heuristic used when no reliable speed data is available.
DEFAULT_FALLBACK_MINUTES_PER_PERCENT = 3.0

# State categories used for interpretation of Home Assistant sensors.
UNKNOWN_STATES = {"unknown", "unavailable", None}
CHARGING_STATES = {"charging", "on", "true"}
DISCHARGING_STATES = {"discharging", "off", "false"}
FULL_STATES = {"full", "complete", "done"}

# Service names exposed by the integration.
SERVICE_FORCE_REFRESH = "force_refresh"
SERVICE_START_CHARGING = "start_charging"
SERVICE_STOP_CHARGING = "stop_charging"
SERVICE_AUTO_MANAGE = "auto_manage"
SERVICE_LOAD_MODEL = "load_model"
