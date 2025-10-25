# Smart Charger for Home Assistant

  <img alt="Smart Charger logo" src="https://raw.githubusercontent.com/enkama/smart_charger/master/logo/icon.png">

Smart Charger is a small Home Assistant custom integration that helps schedule and control charging for battery-powered devices. It combines live sensor data, alarm times, presence and learned charging performance to start and stop chargers just-in-time so devices reach a target battery level when needed.

This README is a concise guide for installation, configuration, and development.

## Quick start

- Install via HACS (recommended) or copy the `custom_components/smart_charger` folder to `config/custom_components` and restart Home Assistant.
- Install via HACS (recommended):

  1. In HACS open the three-dot menu (top-right) and choose "Custom repositories".
  2. Add repository URL: `https://github.com/enkama/smart_charger` and set type to `Integration`.
  3. Click "ADD", then find and install "Smart Charger" in HACS → Integrations.
  4. Restart Home Assistant after installation.

  After installing via HACS (or manually), add the integration via Settings → Integrations → Add Integration → Smart Charger. Create one entry per device you want to manage.

## Highlights

- Predictive charging based on alarm schedules and learned charge speed
- Per-device precharge thresholds, hysteresis and countdown windows to avoid toggling
 - Per-device automatic-control toggle: a switch entity is created for each configured device so you can enable/disable the automatic management per device (visible on the Integration -> Devices page)
- Adaptive throttles and flip-flop detection to suppress rapid on/off cycles
- Diagnostics surfaced in the UI for tracing decisions and tuning
- Services to control or refresh the coordinator

## Installation

### HACS (recommended)
1. In HACS go to: "Integrations" → three-dot menu → "Custom repositories".
2. Add repository URL: `https://github.com/enkama/smart_charger` and select type `Integration`.
3. Install the integration and restart Home Assistant.

### Manual
1. Copy the `custom_components/smart_charger` directory into your Home Assistant `config/custom_components` folder.
2. Ensure `translations` and all python files are present.
3. Restart Home Assistant.

## Configuration

Smart Charger uses a config flow. Add one entry per device you want the integration to manage.

Per-device settings typically include:

- `battery_sensor` — sensor reporting battery percentage
- `charger_switch` — entity to toggle charging (switch)
- Optional: `charging_sensor`, `presence_sensor`, `avg_speed_sensor`
- `target_level`, `min_level`, `precharge_level` and associated margins
- Per-device `switch_throttle_seconds` and `switch_confirmation_count` to avoid rapid toggles
- Optional alarm entities (single or per-weekday) for scheduled readiness

Most settings are editable via the entry's *Configure* → *Advanced settings* UI.

### Per-device automatic control toggle

When you add a device entry the integration now creates a per-device toggle entity named
"Smart Charger <Device Name> Auto Control". This toggle is persisted in the integration entry
options and is visible on the Integration -> Devices page (select the Smart Charger device)
so you can enable or disable automated control for each device independently.

### Anti-flap protections

To avoid rapid precharge toggles the integration supports two complementary protections:

- `precharge_cooldown_minutes` — minimum minutes after a precharge pause before re-allowing precharge
- `precharge_min_drop_percent` — require the battery to drop by this percent after release before re-activating precharge

Both protections can be used together; sensible defaults are conservative but adjustable in the options flow.

### Configuration example

Quick UI steps to add a device entry:

1. Settings → Integrations → Add Integration → Smart Charger
2. Fill `name`, choose `battery_sensor` (percentage sensor) and `charger_switch` (switch)
3. Optionally add `charging_sensor` / `presence_sensor` and adjust targets/margins

Minimal `data` shape (for reference only — use the UI to create entries):

```json
{
  "devices": [
    {
      "name": "My Device",
      "battery_sensor": "sensor.my_device_battery",
      "charger_switch": "switch.my_device_charger",
      "target_level": 95.0,
      "min_level": 30.0,
      "precharge_level": 50.0
    }
  ]
}
```

The UI will expose many more optional fields (precharge margins, cooldown, throttles). Use the Advanced settings for per-device tuning.

## Services

The integration registers the following services in the `smart_charger` domain:

- `force_refresh` — request an immediate update
- `start_charging` — force a device into charging state
- `stop_charging` — stop charging for a device
- `auto_manage` — run the coordinator decision logic once (can be used in automations)
- `load_model` — reload learned charge-speed data from storage

See `custom_components/smart_charger/services.yaml` for exact schemas and optional parameters.

## Diagnostics & tuning

Open the integration entry in the UI to view diagnostics. The diagnostics show effective thresholds and the coordinator's recent decisions so you can fine-tune margins, throttles and adaptive settings.

Recommended starting values (tweak as needed):

- `precharge_margin_on` (release margin): 1.5%
- `precharge_margin_off` (resume margin): 0.5%
- `precharge_countdown_window`: 5%
- `precharge_min_drop_percent`: 1.0% (prevents immediate re-activation)
- `precharge_cooldown_minutes`: 10 (minutes)
- `switch_throttle_seconds`: 30
- `switch_confirmation_count`: 2

## Troubleshooting tips

- If the entity dropdowns show `[object Object]` in the UI, update to the branch `fix/review-suggestions-friendly-entities` — recent fixes change the options to use structured SelectSelectors so labels render correctly.
- If chargers toggle too frequently, increase `switch_throttle_seconds` and/or `switch_confirmation_count` and review `precharge_*` margins.
- Use the diagnostics page on the integration entry to inspect the coordinator's recent decisions.

## Development

- Run tests with `pytest` in the repository root (project includes unit tests for coordinator helpers).
- Keep `black`, `flake8`, and `mypy` happy; the project includes lint scripts in `scripts/`.
- Use the Home Assistant developer docs for guidance when modifying config flows or registries.

If you add new options or translations, update `strings.json` and the `translations/` files.

## Contributing

Contributions are welcome via pull requests. Please include tests for behavior changes and keep changes small and focused.

## License

Smart Charger is licensed under the MIT License. See `LICENSE` for details.
