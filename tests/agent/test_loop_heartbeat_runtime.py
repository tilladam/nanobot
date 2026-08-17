from pathlib import Path
from unittest.mock import MagicMock, patch

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config


def _make_loop(tmp_path: Path, *, heartbeat_model: str | None = None) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=8000,
        heartbeat_model=heartbeat_model,
    )


def test_heartbeat_runtime_is_none_by_default(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    assert loop.heartbeat_runtime() is None


def test_heartbeat_runtime_overrides_model_without_changing_default(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, heartbeat_model="cheap-model")

    runtime = loop.heartbeat_runtime()

    assert runtime is not None
    assert runtime.model == "cheap-model"
    assert runtime.provider is loop.provider
    assert loop.model == "test-model"


def test_from_config_wires_heartbeat_model(tmp_path: Path) -> None:
    config = Config.model_validate({
        "agents": {"defaults": {"model": "openai/gpt-4.1", "workspace": str(tmp_path)}},
        "gateway": {"heartbeat": {"model": "openai/gpt-4.1-mini"}},
    })
    provider = MagicMock()
    provider.get_default_model.return_value = "openai/gpt-4.1"

    with patch("nanobot.providers.factory.make_provider", return_value=provider):
        loop = AgentLoop.from_config(config, tool_registry=ToolRegistry())

    assert loop.heartbeat_model == "openai/gpt-4.1-mini"
