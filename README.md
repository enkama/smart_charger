# Smart Charger for Home Assistant

  <img alt="Smart Charger logo" src="https://raw.githubusercontent.com/enkama/smart_charger/master/logo/icon.png">

Smart Charger is a custom integration for Home Assistant that orchestrates predictive charging for battery powered devices. It monitors battery levels, presence, alarms, and historical charge performance to trigger just-in-time charging cycles. The integration exposes rich diagnostics, helper sensors, and a set of services that can be automated from Home Assistant automations and scripts.

## Features

- Track one or more devices with dedicated battery and charger entities
- Predict required charging windows based on alarm schedules and presence signals
- Automatically start, stop, or suggest charging via services or notifications
- Maintain historical charging performance for smarter future suggestions
- React to fresh charge sessions instantly while decaying older data over time
- Provide diagnostics data and runtime insights through the Home Assistant UI

## Installation

### HACS (recommended)

This integration can be added to HACS as a custom repository.

HACS Menu

1. Click on the 3 dots in the top right corner.
2. Select "Custom repositories".
3. Add `https://github.com/enkama/smart_charger` as the repository URL.
4. Select the `Integrations` type.
5. Click the "ADD" button.
6. Install **Smart Charger** from the custom repositories list and restart Home Assistant.

### Manual installation

If you do not use HACS you can install the integration by copying the folder manually:

1. Download the latest release archive (or use the GitHub *Raw* view when saving individual files).
2. Copy the entire `custom_components/smart_charger` directory from the archive into your Home Assistant `config/custom_components` directory. Create the folder structure if it does not exist yet.
3. Ensure all files and subfolders are present, including the translation resources:

```
config
  custom_components
    smart_charger
      translations
        *.json
      __init__.py
      config_flow.py
      const.py
      coordinator.py
      diagnostics.py
      learning.py
      manifest.json
      sensor.py
      services.py
      services.yaml
```

4. Restart Home Assistant to load the integration.

Repeat these steps whenever a new release is published.

## Configuration

The integration provides a config flow. Each device entry supports the following options:

- Battery level sensor (percentage)
- Charger switch entity to toggle charging
- Optional charging state, presence, and average speed sensors (speed should report percentage gained per hour)
- Predictive mode with target, minimum, and precharge levels
- Tune precharge hysteresis (release/resume margins) and SmartStart finish buffers
- Configure a precharge release countdown window to avoid charger flapping
- Configure a per-device switch throttle (seconds) and confirmation count to avoid rapid toggles:
  - `switch_throttle_seconds` (default 30s): minimum seconds between issuing switch commands for the same charger entity.
  - `switch_confirmation_count` (default 2): number of consecutive coordinator evaluations that must request the same desired state before the integration will send the switch command. This helps prevent toggles caused by short-lived sensor noise.
- Alarm entities that define desired ready times per weekday
- Notification targets and thresholds for charging suggestions

> **Tip:** Provide either a learning source or an average speed sensor so the coordinator can estimate how long charging takes. Newly finished charge sessions from the last four hours are treated as ground-truth speeds, while older samples decay with a ~12 hour half-life. If neither source is available the integration falls back to a conservative 24 hour charging window.

You can adjust how long those "recent" samples remain authoritative from the options menu under **Advanced settings**. Increasing the window helps when you charge infrequently, while reducing it favors more immediate behaviour at the cost of relying on fallback statistics sooner.

You can build the average speed sensor with Home Assistant helpers if your device does not expose one directly. For example, create a [Template Sensor](https://www.home-assistant.io/integrations/template/) that measures the delta of your battery percentage over time, then wrap it with the [Statistics Sensor](https://www.home-assistant.io/integrations/statistics/) using a mean of the last few charge sessions. Aim to express the result in `%/h`, which Smart Charger interprets as the fallback rate whenever no learned data is available.

You can revisit the options via the entry's *Configure* button to edit or remove devices at any time.

Diagnostics expose both the configured and effective margins so you can confirm what the coordinator currently applies.

### Precharge hysteresis & retry behavior

- The coordinator keeps chargers running using a hysteresis window: by default it releases after reaching the precharge level plus `1.5%` and resumes if the level falls `0.5%` below the target. Both margins, as well as the SmartStart finish buffer (`2%`), can be adjusted in the options flow per device.
- A configurable release countdown window (default `5%` above the precharge level) prevents rapid toggling: once the battery enters that band the charger continues for 10 minutes before pausing, unless the level drops below the band again.
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
