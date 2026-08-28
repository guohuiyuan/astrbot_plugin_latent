from __future__ import annotations

import httpx
import pytest

from latent.api import (
    GenerationJob,
    LatentAPIClient,
    LatentAPIError,
    normalize_tags,
)


def test_normalize_tags_variants():
    assert normalize_tags("1girl, long hair") == ["1girl", "long_hair"]
    assert normalize_tags("1girl; sunset") == ["1girl", "sunset"]
    assert normalize_tags(["1girl", " long hair "]) == ["1girl", "long_hair"]
    assert normalize_tags(" , ") == []


def test_generation_job_terminal():
    job = GenerationJob.from_dict(
        {
            "id": "abc",
            "status": "succeeded",
            "prompt": "1girl",
            "seed": 1,
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "createdAt": "2026-01-01T00:00:00Z",
            "artworkId": "art-1",
        }
    )
    assert job.is_terminal() is True
    assert job.artwork_id == "art-1"

    queued = GenerationJob.from_dict(
        {
            "id": "abc",
            "status": "running",
            "prompt": "1girl",
            "seed": 1,
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "createdAt": "2026-01-01T00:00:00Z",
        }
    )
    assert queued.is_terminal() is False


@pytest.mark.asyncio
async def test_resolve_image_builds_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["params"] = dict(request.url.params.multi_items())
        payload = {
            "id": "uuid-1",
            "imageUrl": "https://latent.moe/api/media/uuid-1?size=preview",
            "artworkUrl": "https://latent.moe/art/uuid-1",
            "width": 920,
            "height": 1536,
            "source": "comfyui",
            "model": "test-model",
            "matchedTags": ["1girl", "long_hair"],
            "totalTagCount": 2,
            "rank": 1,
            "viewCount": 10,
            "nsfw": False,
        }
        return httpx.Response(200, json={"data": payload})

    transport = httpx.MockTransport(handler)
    async with LatentAPIClient(api_key="key", transport=transport) as client:
        hit = await client.resolve_image(["1girl", "long hair"], rank=1, size="preview")

    assert hit.id == "uuid-1"
    assert hit.source == "comfyui"
    assert "tag=1girl" in seen["url"]
    assert "tag=long_hair" in seen["url"]
    assert hit.matched_tags == ["1girl", "long_hair"]


@pytest.mark.asyncio
async def test_enqueue_generation_posts_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content.decode()
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            202,
            json={
                "id": "job-1",
                "status": "queued",
                "prompt": "1girl, solo",
                "seed": 123,
                "width": 920,
                "height": 1536,
                "steps": 12,
                "createdAt": "2026-01-01T00:00:00Z",
                "queuePosition": 1,
            },
        )

    transport = httpx.MockTransport(handler)
    async with LatentAPIClient(api_key="lat_sk_x", transport=transport) as client:
        job = await client.enqueue_generation(
            "1girl, solo",
            negative_prompt="bad anatomy",
            resolution="portrait",
            steps=12,
            sampler="euler",
            scheduler="normal",
        )

    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer lat_sk_x"
    assert '"prompt":"1girl, solo"' in seen["body"]
    assert '"negativePrompt":"bad anatomy"' in seen["body"]
    assert job.status == "queued"
    assert job.queue_position == 1


@pytest.mark.asyncio
async def test_fetch_media_returns_bytes():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/media/art-1" in str(request.url)
        return httpx.Response(200, content=b"image-bytes")

    transport = httpx.MockTransport(handler)
    async with LatentAPIClient(api_key="key", transport=transport) as client:
        data = await client.fetch_media("art-1", size="preview")
    assert data == b"image-bytes"


@pytest.mark.asyncio
async def test_error_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": "invalid_parameter", "message": "resolution out of bounds"},
        )

    transport = httpx.MockTransport(handler)
    async with LatentAPIClient(api_key="key", transport=transport) as client:
        with pytest.raises(LatentAPIError) as exc:
            await client.enqueue_generation("1girl", resolution="tiny")
    assert exc.value.status_code == 422
    assert exc.value.code == "invalid_parameter"
