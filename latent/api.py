from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


DEFAULT_API_BASE = "https://latent.moe"
DEFAULT_TIMEOUT = 30.0


class LatentAPIError(Exception):
    """Raised when the Latent.moe API returns a non-success response."""

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class ResolverResponse:
    """Resolution of a public image by prompt tags."""

    id: str
    image_url: str
    artwork_url: str
    width: int
    height: int
    source: str
    matched_tags: list[str]
    total_tag_count: int
    rank: int
    view_count: int
    model: str | None = None
    nsfw: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolverResponse":
        return cls(
            id=str(data["id"]),
            image_url=str(data["imageUrl"]),
            artwork_url=str(data["artworkUrl"]),
            width=int(data["width"]),
            height=int(data["height"]),
            source=str(data["source"]),
            matched_tags=list(data.get("matchedTags") or []),
            total_tag_count=int(data.get("totalTagCount") or 0),
            rank=int(data.get("rank") or 1),
            view_count=int(data.get("viewCount") or 0),
            model=data.get("model"),
            nsfw=bool(data.get("nsfw", False)),
        )


@dataclass
class GenerationJob:
    """A generation job as returned by the Latent.moe API."""

    id: str
    status: str
    prompt: str
    seed: int
    width: int
    height: int
    steps: int
    created_at: str
    negative_prompt: str | None = None
    sampler: str | None = None
    scheduler: str | None = None
    resolution: str | None = None
    progress: int | None = None
    error_code: str | None = None
    artwork_id: str | None = None
    visibility: str | None = None
    pending: bool = False
    queue_position: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationJob":
        return cls(
            id=str(data["id"]),
            status=str(data["status"]),
            prompt=str(data.get("prompt") or ""),
            seed=int(data.get("seed") or 0),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            steps=int(data.get("steps") or 0),
            created_at=str(data.get("createdAt") or ""),
            negative_prompt=data.get("negativePrompt"),
            sampler=data.get("sampler"),
            scheduler=data.get("scheduler"),
            resolution=data.get("resolution"),
            progress=data.get("progress"),
            error_code=data.get("errorCode"),
            artwork_id=data.get("artworkId"),
            visibility=data.get("visibility"),
            pending=bool(data.get("pending", False)),
            queue_position=data.get("queuePosition"),
        )

    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}


@dataclass
class ArtworkMeta:
    """Additional metadata for an owned artwork, used to enrich generation results."""

    id: str
    generator: str | None = None
    model: str | None = None
    mime: str | None = None
    file_size: int | None = None
    tag_count: int | None = None
    nsfw: bool = False
    moderation_status: str | None = None
    artwork_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtworkMeta":
        return cls(
            id=str(data["id"]),
            generator=data.get("generator"),
            model=data.get("model"),
            mime=data.get("mime"),
            file_size=data.get("fileSize"),
            tag_count=data.get("tagCount"),
            nsfw=bool(data.get("nsfw", False)),
            moderation_status=data.get("moderationStatus"),
            artwork_url=data.get("artworkUrl"),
        )


class LatentAPIClient:
    """Async client for the Latent.moe API.

    Only the endpoints used by the plugin are implemented. A key is only
    attached to requests that the API protects with the ``ApiKeyBearer`` scheme.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        headers = {"User-Agent": "astrbot-plugin-latent"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    async def __aenter__(self) -> "LatentAPIClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        code = None
        message = f"Latent API request failed: HTTP {status}"
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    code = error.get("code")
                    message = error.get("message") or message
                elif isinstance(error, str):
                    code = error
                    message = body.get("message") or message
        except ValueError:
            pass
        raise LatentAPIError(message, status_code=status, code=code)

    async def resolve_image(
        self,
        tags: list[str] | str,
        *,
        rank: int = 1,
        size: str = "preview",
        source: str | None = None,
        model: str | None = None,
    ) -> ResolverResponse:
        """Resolve a public, safe-for-work image that matches every tag."""
        tag_values = normalize_tags(tags)
        if not tag_values:
            raise LatentAPIError("At least one tag is required to resolve an image", code="missing_tags")
        params: dict[str, Any] = {"format": "json", "rank": rank, "size": size}
        params["tag"] = tag_values
        if source:
            params["source"] = source
        if model:
            params["model"] = model
        response = await self._client.get("/api/v1/images/resolve", params=params)
        self._raise_for_error(response)
        payload = response.json()
        return ResolverResponse.from_dict(payload["data"])

    async def check_generation_capacity(self) -> dict[str, Any]:
        """Return worker availability and queue depth."""
        response = await self._client.get("/api/generate/status")
        self._raise_for_error(response)
        return response.json()

    async def enqueue_generation(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        seed: int | None = None,
        resolution: str = "portrait",
        steps: int = 12,
        sampler: str = "euler",
        scheduler: str = "normal",
    ) -> GenerationJob:
        """Queue a generation job. The job runs asynchronously."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "resolution": resolution,
            "steps": steps,
            "sampler": sampler,
            "scheduler": scheduler,
        }
        if negative_prompt:
            payload["negativePrompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed
        response = await self._client.post("/api/generate", json=payload)
        self._raise_for_error(response)
        return GenerationJob.from_dict(response.json())

    async def get_generation(self, job_id: str) -> GenerationJob:
        """Fetch the current state of a generation job."""
        response = await self._client.get(f"/api/generate/{job_id}")
        self._raise_for_error(response)
        return GenerationJob.from_dict(response.json())

    async def cancel_generation(self, job_id: str) -> bool:
        """Cancel a queued/leased generation. Returns True when cancelled."""
        response = await self._client.post(f"/api/generate/{job_id}/cancel")
        self._raise_for_error(response)
        payload = response.json()
        return bool(payload.get("ok", False))

    async def get_artwork_meta(self, artwork_id: str) -> ArtworkMeta | None:
        """Look up an owned artwork's generator/model metadata by id.

        The API exposes owned artwork listing rather than a single-artwork GET,
        so this pages through the caller's newest artworks and returns the first
        match. Returns ``None`` when the artwork cannot be found.
        """
        cursor: str | None = None
        for _ in range(10):
            params: dict[str, Any] = {"mine": "1", "limit": "100"}
            if cursor:
                params["cursor"] = cursor
            response = await self._client.get("/api/v1/artworks", params=params)
            self._raise_for_error(response)
            payload = response.json()
            items = payload.get("data") or []
            for item in items:
                if str(item.get("id")) == artwork_id:
                    return ArtworkMeta.from_dict(item)
            cursor = payload.get("nextCursor")
            if not cursor:
                break
        return None

    async def fetch_media(self, artwork_id: str, size: str | None = None) -> bytes:
        """Download the bytes of an owned or public artwork.

        Public artwork URLs are served plain; owned/private artworks require the
        bearer key and use the ``/media/{id}`` route.
        """
        path = f"/media/{artwork_id}"
        if size:
            path = f"{path}?size={size}"
        response = await self._client.get(path)
        self._raise_for_error(response)
        return response.content

    @staticmethod
    def media_url(artwork_id: str, size: str | None = None) -> str:
        """The absolute URL used to fetch an artwork by id."""
        suffix = f"?size={size}" if size else ""
        return f"{DEFAULT_API_BASE}/media/{artwork_id}{suffix}"


def normalize_tags(tags: list[str] | str | None) -> list[str]:
    """Normalize a tags/tag-vector value into a clean list of Danbooru tags."""
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = tags
    else:
        raw = ",".join(tags)
    # Commas, semicolons, and spaces all separate tags; underscores are kept.
    parts = [p for p in raw.replace(";", ",").split(",") if p]
    normalized: list[str] = []
    for part in parts:
        tag = part.strip().replace(" ", "_")
        tag = tag.strip("_")
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized
