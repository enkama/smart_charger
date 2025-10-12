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

Not all repositories will work in HACS, since HACS still needs the repository to have a known structure.

If you want to add this custom repository to HACS, do the following:

overview_menu

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
- Optional charging state, presence, and average speed sensors
- Predictive mode with target, minimum, and precharge levels
- Alarm entities that define desired ready times per weekday
- Notification targets and thresholds for charging suggestions

You can revisit the options via the entry's *Configure* button to edit or remove devices at any time.

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
