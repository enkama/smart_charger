## Contributing & Developer notes

This file contains short notes useful when developing or debugging the
Smart Charger custom integration.

### Quieting coordinator logs during CI/tests

The coordinator uses an internal logging adapter that demotes many
informational and non-actionable warning messages to DEBUG. This keeps
test runs and CI logs much quieter while preserving DEBUG and exception
output for debugging.

If you need to see the original INFO/WARNING messages for local
debugging, you can temporarily bypass the adapter in one of two ways:

1) Force the module to use the real logger at runtime (quick, temporary).

```python
# in a debugging session or test setup, e.g. tests/conftest.py
from custom_components.smart_charger import coordinator

# Restore the real logger so .info/.warning are visible again
coordinator._LOGGER = coordinator._REAL_LOGGER
```

2) Adjust Python's logging level to show debug output (recommended when
you want all diagnostic output without mutating module state):

```python
import logging
logging.getLogger('custom_components.smart_charger').setLevel(logging.DEBUG)
```

Notes:
- The adapter still preserves explicit calls that use `_LOGGER.log(level, ...)`
  (used by the integration to emit intentional warnings/errors) so truly
  actionable messages remain visible unless you demote them manually.
- Prefer method (2) for transient debugging since it doesn't mutate module
  attributes and works across processes that import the component.
