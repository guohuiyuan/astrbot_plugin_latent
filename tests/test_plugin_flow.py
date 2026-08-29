from __future__ import annotations

import asyncio
import json
import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.plugins.astrbot_plugin_latent.latent.api import (
    ArtworkMeta,
    GenerationJob,
    LatentAPIError,
    ResolverResponse,
)
from data.plugins.astrbot_plugin_latent.latent.metadata import parse_png_metadata
from data.plugins.astrbot_plugin_latent.main import LatentPlugin


class FakeClient:
    def __init__(self, succeeded: bool = True):
        self.enqueued = 0
        self.succeeded = succeeded
        self.last_payload: dict[str, Any] = {}

    async def check_generation_capacity(self) -> dict[str, Any]:
        return {"workersOnline": 2, "queued": 0}

    async def enqueue_generation(self, prompt: str, **kwargs: Any) -> GenerationJob:
        self.enqueued += 1
        self.last_payload = {"prompt": prompt, **kwargs}
        return GenerationJob(
            id=f"job-{self.enqueued}",
            status="queued",
            prompt=prompt,
            seed=42,
            width=920,
            height=1536,
            steps=12,
            created_at="2026-01-01T00:00:00Z",
            queue_position=1,
        )

    async def get_generation(self, job_id: str) -> GenerationJob:
        status = "succeeded" if self.succeeded else "failed"
        return GenerationJob(
            id=job_id,
            status=status,
            prompt="1girl, solo",
            seed=42,
            width=920,
            height=1536,
            steps=12,
            created_at="2026-01-01T00:00:00Z",
            artwork_id="art-1" if self.succeeded else None,
            error_code=None if self.succeeded else "timeout",
            sampler="euler",
            scheduler="normal",
        )

    async def fetch_media(self, artwork_id: str, size: str) -> bytes:
        return b"png-bytes"

    async def get_artwork_meta(self, artwork_id: str) -> ArtworkMeta:
        return ArtworkMeta(
            id=artwork_id,
            generator="comfyui",
            model=None,
            file_size=123456,
            tag_count=5,
            artwork_url=f"https://latent.moe/art/{artwork_id}",
        )

    async def aclose(self) -> None:
        return None


class FakeSequentialClient(FakeClient):
    """Fake that tracks count and seed progression across a sequential batch."""

    async def enqueue_generation(self, prompt: str, **kwargs: Any) -> GenerationJob:
        self.enqueued += 1
        self.last_payload = {"prompt": prompt, **kwargs}
        return GenerationJob(
            id=f"job-{self.enqueued}",
            status="queued",
            prompt=prompt,
            seed=int(kwargs.get("seed") or 0),
            width=920,
            height=1536,
            steps=int(kwargs.get("steps") or 12),
            created_at="2026-01-01T00:00:00Z",
            queue_position=1,
        )

    async def get_generation(self, job_id: str) -> GenerationJob:
        return GenerationJob(
            id=job_id,
            status="succeeded",
            prompt="1girl, solo",
            seed=self.last_payload.get("seed", 0),
            width=920,
            height=1536,
            steps=12,
            created_at="2026-01-01T00:00:00Z",
            artwork_id=f"art-{job_id}",
            sampler="euler",
            scheduler="normal",
        )

    async def fetch_media(self, artwork_id: str, size: str) -> bytes:
        return f"bytes:{artwork_id}".encode()


class FakeEvent:
    def __init__(self):
        self.sent: list[Any] = []
        self.llm_disabled = False
        self.message_str = "/生图 1girl, solo"
        self.message_id = "123"
        self._self_id = "123456"
        self.reacted: str | None = None
        self.unified_msg_origin = "origin"
        self.bot = SimpleNamespace(call_action=AsyncMock(return_value=None))

    def get_self_id(self) -> str:
        return self._self_id

    def should_call_llm(self, value: bool) -> None:
        self.llm_disabled = value

    async def react(self, emoji: str) -> None:
        self.reacted = emoji

    async def send(self, chain: Any) -> None:
        self.sent.append(chain)

    def plain_result(self, text: str) -> dict[str, Any]:
        return {"type": "plain", "text": text}

    def chain_result(self, chain: list[Any]) -> dict[str, Any]:
        return {"type": "chain", "chain": chain}


def make_plugin(fake_client: FakeClient) -> LatentPlugin:
    context = AsyncMock()
    plugin = LatentPlugin(context, {"apiKey": "lat_sk_test"})
    plugin.client = fake_client  # type: ignore[assignment]
    return plugin


@pytest.mark.asyncio
async def test_generate_and_send_success():
    fake = FakeClient(succeeded=True)
    plugin = make_plugin(fake)
    event = FakeEvent()

    with patch("asyncio.sleep", new=AsyncMock()):
        results = [r async for r in plugin._generate_and_send(event, ["1girl", "solo"], {"count": "1"})]

    assert fake.enqueued == 1
    assert fake.last_payload["prompt"] == "1girl, solo"
    assert fake.last_payload["resolution"] == "portrait"
    assert len(event.sent) == 0  # no intermediate status messages
    chain = results[-1]
    assert chain["type"] == "chain"
    # Image-first composite: the image is the leading component, params follow.
    from astrbot.core.message.components import Image, Plain

    assert isinstance(chain["chain"][0], Image)
    assert getattr(chain["chain"][0], "file", "").startswith("base64://")
    assert isinstance(chain["chain"][1], Plain)
    params_text = chain["chain"][1].text
    assert "耗时" in params_text
    assert "标签: 1girl, solo" in params_text
    assert "Seed: 42" in params_text
    assert "生成器: comfyui" in params_text
    assert "NSFW:" in params_text


@pytest.mark.asyncio
async def test_generate_and_send_failure_reports():
    fake = FakeClient(succeeded=False)
    plugin = make_plugin(fake)
    event = FakeEvent()

    with patch("asyncio.sleep", new=AsyncMock()):
        results = [r async for r in plugin._generate_and_send(event, ["1girl"], {})]

    assert any(r.get("type") == "plain" and "timeout" in r.get("text", "") for r in results)


@pytest.mark.asyncio
async def test_generate_and_send_no_workers():
    fake = FakeClient(succeeded=True)
    fake.check_generation_capacity = AsyncMock(return_value={"workersOnline": 0, "queued": 5})
    plugin = make_plugin(fake)
    event = FakeEvent()

    results = [r async for r in plugin._generate_and_send(event, ["1girl"], {})]
    assert fake.enqueued == 0
    assert any(r.get("type") == "plain" and "Worker" in r.get("text", "") for r in results)


@pytest.mark.asyncio
async def test_to_danbooru_tags_uses_llm():
    context = AsyncMock()
    context.get_current_chat_provider_id = AsyncMock(return_value="91/glm-5.2")
    context.get_config = MagicMock(
        return_value={"provider_settings": {"fallback_chat_models": []}}
    )
    context.get_all_providers = MagicMock(return_value=[])
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(completion_text="masterpiece, best quality, 1girl, long hair")
    )
    plugin = LatentPlugin(context, {"apiKey": "lat_sk_test"})
    event = SimpleNamespace(unified_msg_origin="origin")

    tags = await plugin._to_danbooru_tags(event, "一个黑发少女")  # type: ignore[arg-type]

    assert tags == ["masterpiece", "best_quality", "1girl", "long_hair"]
    args = context.llm_generate.await_args.kwargs
    assert args["chat_provider_id"] == "91/glm-5.2"
    assert args["prompt"] == "一个黑发少女"
    assert "Danbooru" in args["system_prompt"]


@pytest.mark.asyncio
async def test_to_danbooru_tags_falls_back_to_next_provider():
    context = AsyncMock()
    context.get_current_chat_provider_id = AsyncMock(return_value="91/glm-5.2")
    context.get_config = MagicMock(
        return_value={
            "provider_settings": {
                "fallback_chat_models": ["91/grok-4.6", "91/hy3"],
            }
        }
    )
    context.get_all_providers = MagicMock(return_value=[])

    calls: list[str] = []

    async def llm_mock(*, chat_provider_id: str, **kwargs: Any) -> SimpleNamespace:
        calls.append(chat_provider_id)
        if chat_provider_id == "91/glm-5.2":
            raise RuntimeError("upstream_error 402")
        return SimpleNamespace(completion_text="masterpiece, best quality, 1girl, long hair")

    context.llm_generate = AsyncMock(side_effect=llm_mock)
    plugin = LatentPlugin(context, {"apiKey": "lat_sk_test"})
    event = SimpleNamespace(unified_msg_origin="origin")

    tags = await plugin._to_danbooru_tags(event, "一个黑发少女")  # type: ignore[arg-type]

    assert tags == ["masterpiece", "best_quality", "1girl", "long_hair"]
    assert calls == ["91/glm-5.2", "91/grok-4.6"]


@pytest.mark.asyncio
async def test_to_danbooru_tags_reports_when_all_providers_fail():
    context = AsyncMock()
    context.get_current_chat_provider_id = AsyncMock(return_value="91/glm-5.2")
    context.get_config = MagicMock(
        return_value={"provider_settings": {"fallback_chat_models": ["91/grok-4.6"]}}
    )
    context.get_all_providers = MagicMock(return_value=[])
    context.llm_generate = AsyncMock(side_effect=RuntimeError("upstream_error 402"))
    plugin = LatentPlugin(context, {"apiKey": "lat_sk_test"})
    event = SimpleNamespace(unified_msg_origin="origin")

    with pytest.raises(LatentAPIError):
        await plugin._to_danbooru_tags(event, "一个黑发少女")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_generate_and_send_batch_is_sequential():
    fake = FakeSequentialClient()
    plugin = make_plugin(fake)
    event = FakeEvent()

    with patch("asyncio.sleep", new=AsyncMock()):
        results = [
            r
            async for r in plugin._generate_and_send(
                event, ["1girl", "solo"], {"count": "2", "seed": "100"}
            )
        ]

    assert fake.enqueued == 2
    chains = [r for r in results if r.get("type") == "chain"]
    assert len(chains) == 2
    assert fake.last_payload["seed"] == 101  # seed increments across the batch
    from astrbot.core.message.components import Image

    assert all(isinstance(c["chain"][0], Image) for c in chains)


@pytest.mark.asyncio
async def test_generate_and_send_handles_concurrency_limit():
    fake = FakeClient(succeeded=True)
    plugin = make_plugin(fake)
    event = FakeEvent()

    async def raiser(prompt: str, **kwargs: Any) -> GenerationJob:
        raise LatentAPIError("Wait for your current image to finish.", status_code=409, code="too_many_active")

    fake.enqueue_generation = raiser  # type: ignore[assignment]

    with patch("asyncio.sleep", new=AsyncMock()):
        results = [
            r async for r in plugin._generate_and_send(event, ["1girl"], {"count": "2"})
        ]

    text = "".join(r.get("text", "") for r in results if r.get("type") == "plain")
    assert fake.enqueued == 0
    assert "已有生图任务在运行" in text


@pytest.mark.asyncio
async def test_search_random_returns_public_artwork():
    fake = FakeClient(succeeded=True)
    fake.resolve_image = AsyncMock(
        return_value=ResolverResponse(
            id="pub-1",
            image_url="https://latent.moe/media/pub-1",
            artwork_url="https://latent.moe/art/pub-1",
            width=1024,
            height=1024,
            source="comfyui",
            matched_tags=["1girl"],
            total_tag_count=5,
            rank=3,
            view_count=42,
            model="some-model",
            nsfw=False,
        )
    )
    plugin = make_plugin(fake)
    event = FakeEvent()
    event.message_str = "/搜图 1girl, solo"

    results = [r async for r in plugin.resolve_by_tags(event)]

    assert event.bot.call_action.await_args.args[0] == "set_msg_emoji_like"
    assert event.bot.call_action.await_args.kwargs["message_id"] == 123
    assert event.bot.call_action.await_args.kwargs["emoji_id"] == "76"
    assert event.bot.call_action.await_args.kwargs["self_id"] == "123456"
    chain = results[-1]
    assert chain["type"] == "chain"
    from astrbot.core.message.components import Image, Plain

    assert isinstance(chain["chain"][0], Image)
    assert isinstance(chain["chain"][1], Plain)
    assert "模型: some-model" in chain["chain"][1].text
    assert "页面:" in chain["chain"][1].text


@pytest.mark.asyncio
async def test_search_with_rank_is_deterministic():
    fake = FakeClient(succeeded=True)
    ranks: list[int] = []

    async def resolve(tags, *, rank, size, source, model):
        ranks.append(rank)
        return ResolverResponse(
            id="pub-2",
            image_url="https://latent.moe/media/pub-2",
            artwork_url="https://latent.moe/art/pub-2",
            width=1024,
            height=1024,
            source="comfyui",
            matched_tags=tags,
            total_tag_count=5,
            rank=rank,
            view_count=7,
        )

    fake.resolve_image = AsyncMock(side_effect=resolve)  # type: ignore[method-assign]
    plugin = make_plugin(fake)
    event = FakeEvent()
    event.message_str = "/搜图 1girl, solo --rank 5"

    results = [r async for r in plugin.resolve_by_tags(event)]

    assert event.bot.call_action.await_args.args[0] == "set_msg_emoji_like"
    assert ranks == [5]
    assert results[-1]["type"] == "chain"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command_method, message",
    [
        ("generate_by_tags", "/生图 1girl, solo"),
        ("generate_by_nl", "/画图 一个黑发少女"),
        ("resolve_by_tags", "/搜图 1girl, solo"),
        ("latent_help", "/latent"),
    ],
)
async def test_every_command_reacts_to_original_message(command_method, message):
    fake = FakeClient(succeeded=True)
    fake.resolve_image = AsyncMock(
        return_value=ResolverResponse(
            id="pub-reaction",
            image_url="https://latent.moe/media/pub-reaction",
            artwork_url="https://latent.moe/art/pub-reaction",
            width=1024,
            height=1024,
            source="comfyui",
            matched_tags=["1girl"],
            total_tag_count=3,
            rank=1,
            view_count=1,
            nsfw=False,
        )
    )
    context = AsyncMock()
    context.get_current_chat_provider_id = AsyncMock(return_value="91/glm-5.2")
    context.get_config = MagicMock(
        return_value={"provider_settings": {"fallback_chat_models": []}}
    )
    context.get_all_providers = MagicMock(return_value=[])
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(completion_text="masterpiece, 1girl, solo")
    )
    plugin = make_plugin(fake)
    plugin.context = context
    event = FakeEvent()
    event.message_str = message

    with patch("asyncio.sleep", new=AsyncMock()):
        _ = [r async for r in getattr(plugin, command_method)(event)]

    assert event.bot.call_action.await_args.args[0] == "set_msg_emoji_like"
    assert event.bot.call_action.await_args.kwargs["message_id"] == 123
    assert event.bot.call_action.await_args.kwargs["emoji_id"] == "76"
    assert event.bot.call_action.await_args.kwargs["self_id"] == "123456"
    assert event.reacted is None  # NapCat 路径优先，不触发 fallback


@pytest.mark.asyncio
async def test_react_falls_back_when_call_action_fails():
    fake = FakeClient(succeeded=True)
    plugin = make_plugin(fake)
    event = FakeEvent()
    event.bot.call_action = AsyncMock(
        side_effect=RuntimeError("no such action")
    )

    await plugin._react_to_message(event)

    assert event.bot.call_action.await_args.args[0] == "set_msg_emoji_like"
    assert event.bot.call_action.await_args.kwargs["message_id"] == 123
    assert event.bot.call_action.await_args.kwargs["self_id"] == "123456"
    assert event.reacted == "👍"


@pytest.mark.asyncio
async def test_react_converts_string_message_id_to_int():
    fake = FakeClient(succeeded=True)
    plugin = make_plugin(fake)
    event = FakeEvent()
    event.message_id = "987654"

    await plugin._react_to_message(event)

    assert event.bot.call_action.await_args.kwargs["message_id"] == 987654
    assert event.bot.call_action.await_args.kwargs["self_id"] == "123456"
    assert event.reacted is None


@pytest.mark.asyncio
async def test_react_reads_message_id_from_message_obj():
    """Real AstrBot events expose message_id on message_obj, not the event."""
    fake = FakeClient(succeeded=True)
    plugin = make_plugin(fake)
    event = FakeEvent()
    del event.message_id
    event.message_obj = SimpleNamespace(message_id="555000")

    await plugin._react_to_message(event)

    assert event.bot.call_action.await_args.args[0] == "set_msg_emoji_like"
    assert event.bot.call_action.await_args.kwargs["message_id"] == 555000
    assert event.bot.call_action.await_args.kwargs["self_id"] == "123456"
    assert event.reacted is None


def _make_png_with_prompt(workflow: dict[str, Any]) -> bytes:
    """Build a minimal PNG carrying a ComfyUI ``prompt`` tEXt chunk."""

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", 0)

    payload = json.dumps(workflow).encode("utf-8")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"tEXt", b"prompt\x00" + payload)
        + chunk(b"IEND", b"")
    )


def test_parse_png_metadata_extracts_comfyui_params():
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "models/miaomiaoHarem_anima8Step10.safetensors"},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 12,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
    }

    info = parse_png_metadata(_make_png_with_prompt(workflow))

    assert info["model"] == "miaomiaoHarem_anima8Step10"
    assert info["cfg"] == 1.0
    assert info["strength"] == 1.0
    assert info["sampler"] == "euler"
    assert info["scheduler"] == "normal"
    assert info["steps"] == 12
    assert info["seed"] == 123


def test_parse_png_metadata_ignores_webp_preview():
    assert parse_png_metadata(b"RIFF\x00\x00\x00\x00WEBP") == {}
    assert parse_png_metadata(b"") == {}


def test_describe_generation_includes_embedded_model_cfg_strength():
    job = GenerationJob(
        id="job-1",
        status="succeeded",
        prompt="1girl, solo",
        seed=42,
        width=920,
        height=1536,
        steps=12,
        created_at="2026-01-01T00:00:00Z",
        artwork_id="art-1",
        sampler="euler",
        scheduler="normal",
        visibility="private",
    )
    meta = ArtworkMeta(
        id="art-1",
        generator="comfyui",
        file_size=1752080,
        tag_count=23,
        artwork_url="https://latent.moe/art/art-1",
    )
    text = LatentPlugin._describe_generation(
        job,
        "1girl, solo",
        8.9,
        meta,
        {"model": "miaomiaoHarem_anima8Step10", "cfg": 1.0, "strength": 1.0},
    )

    assert "模型: miaomiaoHarem_anima8Step10" in text
    assert "CFG比例: 1.0" in text
    assert "强度: 1.0" in text
    assert "耗时: 8.9s" in text
    assert "生成器: comfyui" in text
