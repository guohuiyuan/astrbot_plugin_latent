# astrbot_plugin_latent

接入 [Latent.moe](https://latent.moe/) 的 AstrBot 生图插件。

## 功能

- /生图：用户直接输入 Danbooru 标签，调用生成接口出图。
- /画图：用户用自然语言描述画面，插件通过 LLM 将描述转成 Danbooru 标签后调用生成接口出图。
- /搜图：按 Danbooru 标签匹配网站中已公开的安全 AI 图片，不消耗生成配额。

## 命令

```
/生图 1girl, long hair, sunset            # 标签直出
/画图 一个穿着红色连衣裙的黑发少女          # 自然语言画图
/搜图 1girl, night                          # 查找已公开图片
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

搜图命令额外支持 `--rank`、`--size`、`--source`、`--model`。

## 配置

在 AstrBot 后台插件配置中填入：

- `apiKey`：Latent.moe 设计师 API Key（`lat_sk_` 开头）。
- `apiBase`：默认 `https://latent.moe`。
- `defaultResolution` / `defaultSteps` / `defaultSampler` / `defaultScheduler`：生图默认参数。
- `llmProviderId`：自然语言转标签所用的 Provider ID；留空则使用会话当前模型。

## 说明

- 生成接口每周有独立配额，只有成功出图才消耗；`/搜图` 不消耗配额。
- 生成结果默认只在任务内私有，插件通过 API 下载后以图片消息发送，不依赖 NapCat 与宿主机共享路径。
- 自然语言转标签依赖 AstrBot 已配置可用的对话模型 Provider。
