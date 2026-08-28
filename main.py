from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star

from .latent.api import (
    ArtworkMeta,
    GenerationJob,
    LatentAPIClient,
    LatentAPIError,
    ResolverResponse,
    normalize_tags,
)
from .latent.utils import (
    VALID_RESOLUTIONS,
    VALID_SAMPLERS,
    VALID_SCHEDULERS,
    VALID_SIZES,
    VALID_SOURCES,
    extract_tag_text,
    parse_options,
    resolve_enum,
    strip_command,
    to_int,
    try_nonnegative_int,
)


DEFAULT_API_BASE = "https://latent.moe"

TAG_SYSTEM_PROMPT = (
    "You are an expert Danbooru tagger for anime image generation prompts. "
    "Convert the user's natural-language description into a comma-separated list of Danbooru tags. "
    "Rules: use lowercase English Danbooru tags, replace spaces within a tag with underscores, "
    "always begin with quality tags such as masterpiece, best quality, then add subject, "
    "character, style, and scene tags. Keep the prompt faithful to the description and do not invent "
    "characters or settings that were not requested. Prefer safe-for-work tags unless the user "
    "explicitly requests mature content. Output only the comma-separated tag string; no explanations, "
    "no numbering, no markdown, no quotes."
)


class LatentPlugin(Star):
    """接入 Latent.moe 的 AI 生图插件。

    支持三种命令：
    - /生图 <Danbooru 标签>：直接用标签生成图片。
    - /画图 <自然语言描述>：由 LLM 转成 Danbooru 标签后生成图片。
    - /搜图 <标签>：匹配网站已公开的安全图片。

    生成命令支持 --steps / --resolution / --sampler / --scheduler / --negative / --seed / --count。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}

        api_key = str(self.config.get("apiKey", "") or "").strip()
        api_base = str(self.config.get("apiBase", DEFAULT_API_BASE) or DEFAULT_API_BASE).strip()
        timeout = to_int(self.config.get("timeout"), 30, 5, 120)
        self.client = LatentAPIClient(api_key=api_key, api_base=api_base, timeout=timeout)

        self.default_resolution = resolve_enum(
            self.config.get("defaultResolution"), VALID_RESOLUTIONS, "portrait"
        )
        self.default_steps = to_int(self.config.get("defaultSteps"), 12, 8, 16)
        self.default_sampler = resolve_enum(
            self.config.get("defaultSampler"), VALID_SAMPLERS, "euler"
        )
        self.default_scheduler = resolve_enum(
            self.config.get("defaultScheduler"), VALID_SCHEDULERS, "normal"
        )
        self.max_poll_seconds = to_int(self.config.get("maxPollSeconds"), 300, 30, 1800)
        self.max_count = to_int(self.config.get("maxCount"), 2, 1, 8)
        self.llm_provider_id = str(self.config.get("llmProviderId", "") or "").strip()

    async def initialize(self):
        logger.info(
            "[Latent] 插件已加载，API 端点: %s，当前 provider: %s",
            self.client.api_base,
            self.llm_provider_id or "会话默认",
        )

    async def terminate(self):
        await self.client.aclose()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @filter.command("生图", alias={"生成", "draw", "generate", "tag2img"})
    async def generate_by_tags(self, event: AstrMessageEvent):
        """通过 Danbooru 标签直接生图。"""
        event.should_call_llm(True)
        raw = strip_command(
            event.message_str,
            {"生图", "生成", "draw", "generate", "tag2img"},
        )
        if not raw:
            yield event.plain_result(self._help_text())
            return
        prompt, options = parse_options(raw)
        tags = normalize_tags(prompt)
        if not tags:
            yield event.plain_result("请提供至少一个 Danbooru 标签，例如：/生图 1girl, long hair")
            return
        async for result in self._generate_and_send(event, tags, options):
            yield result

    @filter.command("画图", alias={"绘画", "nl", "draw_nl"})
    async def generate_by_nl(self, event: AstrMessageEvent):
        """自然语言描述经由 LLM 转成 Danbooru 标签后生图。"""
        event.should_call_llm(True)
        raw = strip_command(
            event.message_str,
            {"画图", "绘画", "nl", "draw_nl"},
        )
        if not raw:
            yield event.plain_result(self._help_text())
            return
        description, options = parse_options(raw)
        if not description.strip():
            yield event.plain_result("请提供画面描述，例如：/画图 一个穿着红色连衣裙的黑发少女")
            return
        try:
            tags = await self._to_danbooru_tags(event, description)
        except LatentAPIError as exc:
            yield event.plain_result(f"标签转换失败: {exc}")
            return
        except Exception as exc:
            logger.warning("[Latent] 自然语言转标签失败: %s", exc)
            yield event.plain_result(f"标签转换失败: {exc}")
            return
        async for result in self._generate_and_send(event, tags, options):
            yield result

    @filter.command("搜图", alias={"找图", "search", "resolve", "find", "roll", "随机", "抽图", "roll_img"})
    async def resolve_by_tags(self, event: AstrMessageEvent):
        """按 Danbooru 标签随机匹配一张已公开的安全图片。"""
        event.should_call_llm(True)
        raw = strip_command(
            event.message_str,
            {"搜图", "找图", "search", "resolve", "find", "roll", "随机", "抽图", "roll_img"},
        )
        if not raw:
            yield event.plain_result(self._help_text())
            return
        prompt, options = parse_options(raw)
        tags = normalize_tags(prompt)
        if not tags:
            yield event.plain_result("请提供至少一个 Danbooru 标签，例如：/搜图 1girl, sunset")
            return
        size = resolve_enum(options.get("size"), VALID_SIZES, "preview")
        source = resolve_enum(options.get("source"), VALID_SOURCES, None)
        model = str(options.get("model", "") or "").strip() or None
        rank = to_int(options.get("rank"), 1, 1, 1000) if options.get("rank") is not None else None
        try:
            if rank is not None:
                hit = await self.client.resolve_image(
                    tags, rank=rank, size=size, source=source, model=model
                )
            else:
                hit = await self._resolve_random(tags, size=size, source=source, model=model)
        except LatentAPIError as exc:
            if exc.status_code == 404:
                yield event.plain_result("没有找到匹配的公开图片。尝试减少标签或 /生图 直接生成。")
                return
            yield event.plain_result(f"查询失败: {exc}")
            return
        if hit is None:
            yield event.plain_result("没有找到匹配的公开图片。尝试减少标签或 /生图 直接生成。")
            return
        media = await self._fetch_media(hit.id, size=size)
        caption = self._describe_resolve(hit, tags)
        yield event.chain_result([Image.fromBytes(media), Plain(caption)])

    @filter.command("latent", alias={"latent_help", "生图帮助", "画画帮助"})
    async def latent_help(self, event: AstrMessageEvent):
        event.should_call_llm(True)
        yield event.plain_result(self._help_text())

    # ------------------------------------------------------------------
    # Natural language -> Danbooru tags
    # ------------------------------------------------------------------

    async def _to_danbooru_tags(self, event: AstrMessageEvent, description: str) -> str:
        provider_id = self.llm_provider_id or await self.context.get_current_chat_provider_id(
            event.unified_msg_origin
        )
        if not provider_id:
            raise LatentAPIError("未配置可用的 LLM Provider，无法进行自然语言标签转换")
        resp = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=description,
            system_prompt=TAG_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=600,
        )
        raw_text = resp.completion_text or ""
        return extract_tag_text(raw_text)

    # ------------------------------------------------------------------
    # Generation flow
    # ------------------------------------------------------------------

    async def _generate_and_send(self, event: AstrMessageEvent, tags: list[str], options: dict[str, Any]):
        prompt = ", ".join(tags)
        count = to_int(options.get("count"), 1, 1, self.max_count)
        resolution = resolve_enum(options.get("resolution"), VALID_RESOLUTIONS, self.default_resolution)
        steps = to_int(options.get("steps"), self.default_steps, 8, 16)
        sampler = resolve_enum(options.get("sampler"), VALID_SAMPLERS, self.default_sampler)
        scheduler = resolve_enum(options.get("scheduler"), VALID_SCHEDULERS, self.default_scheduler)
        negative = str(options.get("negative", "") or "").strip() or None
        seed = try_nonnegative_int(options.get("seed"))

        try:
            status = await self.client.check_generation_capacity()
        except LatentAPIError as exc:
            yield event.plain_result(f"无法连接生图服务: {exc}")
            return
        workers = int(status.get("workersOnline", 0))
        if workers <= 0:
            yield event.plain_result("暂时没有可用的生图 Worker（队列为空或离线），请稍后再试。")
            return

        # The API allows only GENERATION_CONCURRENCY jobs in flight per key, so
        # batches are produced one image at a time rather than enqueued up front.
        for idx in range(count):
            started = time.perf_counter()
            job_seed = seed + idx if seed is not None else None
            try:
                job = await self.client.enqueue_generation(
                    prompt,
                    negative_prompt=negative,
                    seed=job_seed,
                    resolution=resolution,
                    steps=steps,
                    sampler=sampler,
                    scheduler=scheduler,
                )
            except LatentAPIError as exc:
                message = f"提交生图任务失败: {exc}"
                if exc.status_code == 409 or exc.code == "too_many_active":
                    message = "当前已有生图任务在运行，请等待其完成后再试。"
                elif exc.status_code == 429 or exc.code == "quota_exhausted":
                    message = "本周生图额度已用完，请下周再试。"
                elif exc.status_code == 503 or exc.code == "queue_full":
                    message = "生图队列已满，请稍后再试。"
                yield event.plain_result(message)
                break

            done = await self._poll_jobs(event, [job])
            if not done:
                yield event.plain_result(
                    f"任务 {job.id[:8]} 超时，未在 {self.max_poll_seconds} 秒内完成。"
                )
                continue
            finished = done[0]
            if finished.status != "succeeded" or not finished.artwork_id:
                reason = finished.error_code or finished.status
                yield event.plain_result(f"任务 {finished.id[:8]} 未成功: {reason}")
                continue
            meta = await self._get_meta(finished.artwork_id)
            elapsed = time.perf_counter() - started
            media = await self._fetch_media(finished.artwork_id, size="preview")
            caption = self._describe_generation(finished, prompt, elapsed, meta)
            yield event.chain_result([Image.fromBytes(media), Plain(caption)])

    async def _poll_jobs(
        self,
        event: AstrMessageEvent,
        jobs: list[GenerationJob],
    ) -> list[GenerationJob]:
        async def poll_one(job: GenerationJob) -> GenerationJob:
            deadline = asyncio.get_running_loop().time() + self.max_poll_seconds
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(2)
                try:
                    job = await self.client.get_generation(job.id)
                except LatentAPIError as exc:
                    logger.warning("[Latent] 查询任务 %s 失败: %s", job.id, exc)
                    break
                if job.is_terminal():
                    return job
            return job

        results = await asyncio.gather(*(poll_one(job) for job in jobs))
        return [job for job in results if job.is_terminal()]

    async def _fetch_media(self, artwork_id: str, size: str) -> bytes:
        for attempt in range(3):
            try:
                return await self.client.fetch_media(artwork_id, size=size)
            except LatentAPIError as exc:
                if exc.status_code != 404 or attempt == 2:
                    raise
                await asyncio.sleep(1.5)
        raise LatentAPIError("媒体下载失败")

    async def _get_meta(self, artwork_id: str) -> ArtworkMeta | None:
        """Best-effort lookup of generator/model metadata for an owned artwork."""
        try:
            return await self.client.get_artwork_meta(artwork_id)
        except LatentAPIError as exc:
            logger.warning("[Latent] 获取作品元数据 %s 失败: %s", artwork_id, exc)
            return None

    async def _resolve_random(
        self,
        tags: list[str],
        *,
        size: str = "preview",
        source: str | None = None,
        model: str | None = None,
    ) -> ResolverResponse | None:
        """Resolve a random public SFW artwork that matches every tag.

        The API exposes no shuffle endpoint, so this samples the deterministic
        ``rank`` ordering from the resolver. High ranks are tried first (most
        tags have a large public pool); if the pool is small it falls back to
        low ranks, ending at rank 1 so a match is still returned.
        """
        for _ in range(6):
            rank = random.randint(1, 1000)
            try:
                return await self.client.resolve_image(
                    tags, rank=rank, size=size, source=source, model=model
                )
            except LatentAPIError as exc:
                if exc.status_code != 404:
                    raise
        for rank in random.sample(range(2, 51), 8):
            try:
                return await self.client.resolve_image(
                    tags, rank=rank, size=size, source=source, model=model
                )
            except LatentAPIError as exc:
                if exc.status_code != 404:
                    raise
        try:
            return await self.client.resolve_image(
                tags, rank=1, size=size, source=source, model=model
            )
        except LatentAPIError as exc:
            if exc.status_code == 404:
                return None
            raise

    # ------------------------------------------------------------------
    # Descriptions
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_generation(
        job: GenerationJob,
        prompt: str,
        elapsed: float | None = None,
        meta: ArtworkMeta | None = None,
    ) -> str:
        lines = [
            "已生成图片（"
            + f"{job.width}×{job.height}"
            + (f"，耗时 {elapsed:.1f}s）" if elapsed is not None else "）"),
            f"Prompt: {prompt}",
            f"Seed: {job.seed} · 步数 {job.steps} · {job.resolution or 'portrait'}",
            f"采样器: {job.sampler} · 调度器: {job.scheduler}",
        ]
        if meta:
            detail = []
            if meta.generator:
                detail.append(f"生成器: {meta.generator}")
            if meta.model:
                detail.append(f"模型: {meta.model}")
            if detail:
                lines.append(" · ".join(detail))
        if job.negative_prompt:
            lines.append(f"Negative: {job.negative_prompt}")
        if meta and meta.file_size:
            lines.append(f"文件: {meta.file_size / 1024:.0f} KB · 标签 {meta.tag_count or 0} 个")
        if meta:
            lines.append(f"NSFW: {'是' if meta.nsfw else '否'}")
        if job.visibility:
            lines.append(f"可见性: {job.visibility}")
        if meta and meta.artwork_url:
            lines.append(f"页面: {meta.artwork_url}")
        return "\n".join(lines)

    @staticmethod
    def _describe_resolve(hit: ResolverResponse, tags: list[str]) -> str:
        tags_text = ", ".join(tags)
        lines = [
            f"匹配到公开图片（{hit.width}×{hit.height}，来源 {hit.source}）",
            f"标签: {tags_text}",
            f"命中 {hit.total_tag_count} 个标签，第 {hit.rank} 名 · 浏览 {hit.view_count}",
        ]
        if hit.model:
            lines.append(f"模型: {hit.model}")
        lines.append(f"页面: {hit.artwork_url}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _help_text(self) -> str:
        divider = "—" * 22
        return "\n".join(
            [
                "Latent.moe 生图插件",
                divider,
                "/生图 <Danbooru标签>      直接用标签生图",
                "    例: /生图 1girl, long hair, sunset",
                "/画图 <自然语言描述>      由 LLM 转成标签后生图",
                "    例: /画图 一个穿着红色连衣裙的黑发少女",
                "/搜图 <标签>             随机匹配一张已公开的安全图片",
                "    例: /搜图 1girl, night",
                "    （/roll /随机 /抽图 为别名）",
                divider,
                "可选参数:",
                "  --steps 8-16      采样步数",
                "  --resolution square|portrait|landscape",
                "  --sampler euler|dpmpp_2m|ddim ...",
                "  --scheduler karras|beta|normal|simple|exponential",
                "  --negative \"不想出现的内容\"",
                "  --seed <int>",
                "  --count <1-{max_count}>   顺序批量生成张数",
                "  --rank <int>      指定排名（默认随机）",
                "  --size thumb|preview|original",
                "  --source novelai|sd-webui|comfyui|invokeai",
                "  --model <模型子串>",
            ]
        ).format(max_count=self.max_count)
