# 训练仓库机器人（LoRA）

把仓库 Q&A 数据（`data/sample.jsonl`）用来微调 Qwen2.5-Coder，产出一个小的 LoRA 适配器。

## 为什么不在本机训

本机 i5-4300M（2 核）、无独显、可用内存约 3G，3B 的 QLoRA 会 OOM 且需要数天。训练放在 **Colab 免费 T4（16G 显存）**，30–90 分钟即可。

## 步骤

1. 打开 https://colab.research.google.com → 新建 Notebook → 运行时选择 **T4 GPU**
2. 上传 `colab_train.ipynb` 或把里面的单元粘进去
3. 依次运行：克隆仓库 → 安装依赖 → 训练
4. 最后会下载 `lora-repo-bot.zip`（适配器，几十 MB）

## 本地验证脚本（小模型，可选）

有 GPU 的机器上可以先跑 0.5B 冒烟：

```bash
BASE_MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct EPOCHS=1 python train_lora.py
```

## 常见报错

- `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`
  模型被加载成了 bf16，而 `fp16=True` 的 GradScaler 不能 unscale bf16 梯度（T4 本身也只适合 fp16）。
  脚本按 transformers 版本选 `dtype`/`torch_dtype`，并有 `force_amp_safe()` 兜底。
  还报的话，先确认 Colab 里跑的是最新代码——重跑第一个单元会强制重新 clone。
- `AttributeError: ... torch_dtype ...`：transformers 5 已把参数改名为 `dtype`，脚本里已兼容。

## 训练完怎么用

- Ollama 支持加载 LoRA：`ollama create repo-bot -f Modelfile`，Modelfile 里 `FROM qwen2.5-coder:3b` + `ADAPTER ./lora-repo-bot/adapter_model.safetensors`
- 或 llama.cpp：`--lora` 参数加载适配器
- 把适配器目录放进仓库（几十 MB，可直接提交，不需要 LFS）

## 数据格式

每行一个 JSON：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

数据越多越好，但 3B 上建议几百到几千条、任务单一（这里是"只根据仓库回答"）。
