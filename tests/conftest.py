"""Shared pytest fixtures for Smart Charger tests."""

from __future__ import annotations

import pytest
import logging
from types import SimpleNamespace
from pathlib import Path
import tempfile


def _configure_test_logging() -> None:
    """Reduce log noise from the integration during tests.

    Set the smart_charger logger to WARNING so CI logs focus on failures.
    """
    logging.getLogger("custom_components.smart_charger").setLevel(logging.WARNING)


# Apply conservative logging level immediately so logs emitted during imports
# or early setup don't flood test output.
logging.getLogger("custom_components.smart_charger").setLevel(logging.WARNING)
# Silence the learning module errors that occur when running in the test harness.
logging.getLogger("custom_components.smart_charger.learning").setLevel(logging.CRITICAL)


def pytest_sessionstart(session):
    """Allow sockets early so Home Assistant's event loop can initialise."""
    try:
        import pytest_socket
        from pytest_socket import enable_socket

        enable_socket()
        # pytest-asyncio spins up the loop before pytest-socket gives tests a
        # chance to opt-in, so coerce the disable hook to stay a no-op.

        def _noop_disable_socket(*_args, **_kwargs):
            enable_socket()

        pytest_socket.disable_socket = _noop_disable_socket
    except Exception:  # pragma: no cover
        # If pytest-socket is not installed or behaves differently, skip silently.
        pass


def pytest_configure(config):
    """Ensure socket usage stays enabled even if the plugin sets defaults."""
    try:
        from pytest_socket import enable_socket

        enable_socket()
    except Exception:
        pass
    force = getattr(config.option, "force_enable_socket", None)
    if force is not None:
        config.option.force_enable_socket = True
    disable = getattr(config.option, "disable_socket", None)
    if disable is not None:
        config.option.disable_socket = False
    # Newer versions of pytest-socket cache the derived flags on the config
    # instance, so ensure those mirrors are flipped as well.
    if hasattr(config, "__socket_force_enabled"):
        config.__socket_force_enabled = True
    if hasattr(config, "__socket_disabled"):
        config.__socket_disabled = False


@pytest.fixture(autouse=True)
def _allow_socket(socket_enabled):
    """Keep sockets enabled for every test."""
    yield


@pytest.fixture(autouse=True)
def _test_hass_config_path(hass):
    """Ensure `hass.config.path` exists for storage helpers used by learning.

    The test harness sometimes supplies a SimpleNamespace for hass; provide a
    minimal `config.path` compatible callable that returns a temporary folder
    so storage writes don't raise AttributeError during tests.
    """
    # Configure logging early for all tests
    _configure_test_logging()

    # If hass already has a config with path, leave it alone.
    cfg = getattr(hass, "config", None)
    if cfg and hasattr(cfg, "path"):
        yield
        return

    # Otherwise, attach a lightweight config object with a `path` method.
    tmpdir = Path(tempfile.mkdtemp(prefix="hc_sc_tests_"))

    def _path(*parts: str) -> str:
        return str(tmpdir.joinpath(*parts))

    # Provide a minimal but functional hass.config.path implementation that
    # resolves to our tmpdir. Home Assistant's storage helpers will call
    # `hass.config.path(STORAGE_DIR, key)`; returning tmpdir/storage/key makes
    # those calls succeed without needing to monkeypatch internal Store.
    def _config_path(*parts: str) -> str:
        # preprend our tmpdir to mimic HA layout
        return str(tmpdir.joinpath(*parts))

    hass.config = SimpleNamespace(path=_config_path)

    # Tighter logging: ensure only warnings and errors are shown during tests.
    logging.getLogger().setLevel(logging.WARNING)
    yield
    # No explicit cleanup: pytest will remove temp dirs when the process exits.
