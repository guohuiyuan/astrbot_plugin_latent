from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from data.plugins.astrbot_plugin_latent.latent.api import (
    ArtworkMeta,
    GenerationJob,
    LatentAPIError,
)
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

    def should_call_llm(self, value: bool) -> None:
        self.llm_disabled = value

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
    assert len(event.sent) == 1  # submitted confirmation
    chain = results[-1]
    assert chain["type"] == "chain"
    # Image-first composite: the image is the leading component, params follow.
    from astrbot.core.message.components import Image, Plain

    assert isinstance(chain["chain"][0], Image)
    assert getattr(chain["chain"][0], "file", "").startswith("base64://")
    assert isinstance(chain["chain"][1], Plain)
    params_text = chain["chain"][1].text
    assert "耗时" in params_text
    assert "Prompt: 1girl, solo" in params_text
    assert "Seed: 42" in params_text
    assert "生成器: comfyui" in params_text


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
