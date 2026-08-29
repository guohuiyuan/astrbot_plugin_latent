# astrbot_plugin_latent

接入 [Latent.moe](https://latent.moe/) 的 AstrBot 生图插件。

## 功能

- /生图：用户直接输入 Danbooru 标签，调用生成接口出图。
- /画图：用户用自然语言描述画面，插件通过 LLM 将描述转成 Danbooru 标签后调用生成接口出图。
- /搜图：按 Danbooru 标签随机匹配一张网站中已公开的安全 AI 图片，不消耗生成配额（`/roll`、`/随机`、`/抽图` 为别名）。
- 所有命令收到后都会对原消息贴一个表情回应，用于快速判断插件是否已响应。

## 命令

```
/生图 1girl, long hair, sunset            # 标签直出
/画图 一个穿着红色连衣裙的黑发少女          # 自然语言画图
/搜图 1girl, night                          # 随机匹配一张已公开图片
/roll 1girl, night                          # 同 /搜图（别名）
/latent                                     # 查看帮助
```

生成命令支持可选参数，参数需放在标签/描述之后：

```text
--steps 12                    采样步数 8-16
--resolution portrait         square / portrait / landscape
--sampler dpmpp_2m            euler / euler_ancestral / dpmpp_2m / ddim ...
--scheduler karras            karras / beta / normal / simple / exponential
--negative "..."
--seed 42
--count 2                    顺序生成多张（生成完一张再下一张）
```

搜图命令支持 `--size`、`--source`、`--model`；不传 `--rank` 时在 `1..1000` 内随机，传 `--rank N` 则固定第 N 名。

## 配置

在 AstrBot 后台插件配置中填入：

- `apiKey`：Latent.moe 设计师 API Key（`lat_sk_` 开头）。
- `apiBase`：默认 `https://latent.moe`。
- `defaultResolution` / `defaultSteps` / `defaultSampler` / `defaultScheduler`：生图默认参数。
- `llmProviderId`：自然语言转标签所用的 Provider ID；留空则使用会话当前模型。

## 说明

- 生成接口每周有独立配额，只有成功出图才消耗；`/搜图`（含 `/roll` 等别名）不消耗配额。
- 生成结果默认只在任务内私有，插件通过 API 下载后以图片消息发送，不依赖 NapCat 与宿主机共享路径。
- 自然语言转标签依赖 AstrBot 已配置可用的对话模型 Provider。
- `/画图` 会先使用会话当前模型，失败时按 AstrBot `fallback_chat_models` 顺序自动切换下一个可用模型，全部失败才提示错误。
- 执行命令时不再发送过程提示，最终只返回一条合成消息：图片在上，生成参数/链接在下。
- 生成结果参数包含尺寸、耗时、Prompt、Seed、步数、分辨率、采样器、调度器、生成器、模型、Negative、文件大小、标签数、NSFW、可见性与作品页。
- `/画图` 与 `/生图` 可生成 NSFW 内容（需用户明确要求），结果保持私密并由站方打 NSFW 标记；`/搜图` 只会返回公开安全图片。
