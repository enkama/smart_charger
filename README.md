# Smart Charger for Home Assistant

  <img alt="Smart Charger logo" src="https://raw.githubusercontent.com/enkama/smart_charger/master/logo/icon.png">

Smart Charger is a custom integration for Home Assistant that orchestrates predictive charging for battery powered devices. It monitors battery levels, presence, alarms, and historical charge performance to trigger just-in-time charging cycles. The integration exposes rich diagnostics, helper sensors, and a set of services that can be automated from Home Assistant automations and scripts.

## Features

- Track one or more devices with dedicated battery and charger entities
- Predict required charging windows based on alarm schedules and presence signals
- Automatically start, stop, or suggest charging via services or notifications
- Maintain historical charging performance for smarter future suggestions
- Provide diagnostics data and runtime insights through the Home Assistant UI

## Installation

### HACS (recommended)

This integration can be added to HACS as a custom repository.

**Warning**

If you want to add this custom repository to HACS, do the following:

HACS Menu

1. Click on the 3 dots in the top right corner.
2. Select "Custom repositories".
3. Add `https://github.com/enkama/smart_charger` as the repository URL.
4. Select the `Integrations` type.
5. Click the "ADD" button.
6. Install **Smart Charger** from the custom repositories list and restart Home Assistant.

### Manual updates

If you do not use HACS, repeat the copy step above with every release and restart Home Assistant afterwards.

## Configuration

The integration provides a config flow. Each device entry supports the following options:

- Battery level sensor (percentage)
- Charger switch entity to toggle charging
- Optional charging state, presence, and average speed sensors (speed should report percentage gained per hour)
- Predictive mode with target, minimum, and precharge levels
- Tune precharge hysteresis (release/resume margins) and SmartStart finish buffers
- Alarm entities that define desired ready times per weekday
- Notification targets and thresholds for charging suggestions

> **Tip:** Provide either a learning source or an average speed sensor so the coordinator can estimate how long charging takes. If neither is available the integration falls back to a conservative 24 hour charging window when it cannot determine a usable speed.

You can build the average speed sensor with Home Assistant helpers if your device does not expose one directly. For example, create a [Template Sensor](https://www.home-assistant.io/integrations/template/) that measures the delta of your battery percentage over time, then wrap it with the [Statistics Sensor](https://www.home-assistant.io/integrations/statistics/) using a mean of the last few charge sessions. Aim to express the result in `%/h`, which Smart Charger interprets as the fallback rate whenever no learned data is available.

You can revisit the options via the entry's *Configure* button to edit or remove devices at any time.

Diagnostics expose both the configured and effective margins so you can confirm what the coordinator currently applies.

### Precharge hysteresis & retry behavior

- The coordinator keeps chargers running using a hysteresis window: by default it releases after reaching the precharge level plus `1.5%` and resumes if the level falls `0.5%` below the target. Both margins, as well as the SmartStart finish buffer (`2%`), can be adjusted in the options flow per device.
- Learning session finalization will retry when the battery sensor is temporarily unavailable, using backoff delays of 30s, 90s, and 300s. Diagnostics show the current retry attempt count together with the active delay schedule.
- Coordinator diagnostics now surface the active hysteresis release thresholds, making it easier to verify when and why a charger remains running.

## Services

The integration registers the following domain services:

- `smart_charger.force_refresh` – trigger an immediate update of the coordinator
- `smart_charger.start_charging` – force a device into the charging state
- `smart_charger.stop_charging` – stop active charging
- `smart_charger.auto_manage` – run the predictive scheduling logic manually
- `smart_charger.load_model` – reload the learned charging characteristics from storage

See `custom_components/smart_charger/services.yaml` for the exact service schemas.

## Development

- Validate metadata with `Home Assistant hassfest` and the HACS GitHub action workflow (included in `.github/workflows/validate.yaml`).
- Run integration tests inside a Home Assistant development container if you extend the module.
- Follow the [Home Assistant developer documentation](https://developers.home-assistant.io/) when modifying config flows, coordinators, or entities.

## License

Smart Charger is distributed under the [MIT License](LICENSE) by enkama.

This project is licensed under the MIT License. See `LICENSE` for details.
