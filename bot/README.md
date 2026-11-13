# 仓库机器人（本地模型）

零依赖 JS 机器人：推理走本机模型 API，仓库文件就是它的知识与记忆。

## 用法

```bash
# 只读问答
node bot/bot.mjs "这个仓库是干什么的？"

# 允许改文件
node bot/bot.mjs "在 README 末尾加一行说明" --write
```

## 配置 `bot/config.json`

```json
{
  "api": "http://127.0.0.1:18801/v1",
  "model": "opencode/deepseek-v4.1-flash-free",
  "max_turns": 6,
  "temperature": 0.2
}
```

- `api`：任何 OpenAI 兼容端点（免费桥 / 本地 ollama 均可）
- `model`：模型名
- 也可用环境变量覆盖：`MODEL_API_URL`、`MODEL_NAME`、`BOT_MAX_TURNS`

## 大脑 `bot/brain.md`

机器人的人设、规则和行为准则。改这个文件就等于改它的"脑子"。

## 安全

- 只能读写当前仓库内的路径（越界会报错）
- `write_file` 只有加 `--write` 才启用
- 不要提交 `.env`、`*.pem`（已在 `.gitignore`）
