# SWE-SolveAgent

[English](README.md) | **中文**

**面向 [SWE-bench](https://www.swebench.com/) 的多阶段 LLM 修复智能体：在真实 GitHub Issue 上自动生成补丁。**

输入 bug 报告与 Docker 沙箱中的代码仓库，智能体返回 **unified diff 补丁**。核心入口是 [`src/agent.py`](src/agent.py) 中的 `solve_task`：

```text
issue + repo @ /testbed
        │
        ▼
   localize  →  generate  →  validate  →  refine  →  best patch
```

设计灵感来自 [PatchPilot](https://arxiv.org/abs/2502.02747) / Agentless 一类**规则化多阶段流水线**（固定阶段顺序，而非开放式 agent 循环）。

![SWE-SolveAgent 总览](assets/pipeline.jpg)

---

## 核心能力

| 阶段 | 作用 |
|------|------|
| **Localize** | 分层故障定位：文件 → 函数 / 代码行 |
| **Generate** | 先规划再生成多候选补丁（SEARCH/REPLACE → git diff） |
| **Validate** | 在 Docker 中确定性跑测试并排序（**不消耗 LLM token**） |
| **Refine** | 在 token 预算内根据失败测试迭代优化，返回当前最优 |

- 使用官方 **SWE-bench Docker** 环境
- 模型客户端与具体厂商解耦（默认 DeepSeek，OpenAI 兼容接口）
- 带 **token 预算** 守卫，控制多阶段修复成本

### 效率（样例运行）

在 **5 个任务** 的 SWE-bench Lite 子集上（`deepseek-v3.2`，最终流水线）：

| 指标 | 数值 |
|------|-----:|
| Prompt tokens | 363,485 |
| Completion tokens | 61,087 |
| **Total tokens** | **424,572** |
| **≈ 每任务** | **~84.9k** |

验证 / 排序为确定性 Docker 测试，不占用 LLM token。以上数字反映该规模下的成本画像，**不是**完整榜单成绩。

---

## 架构

实现位于 `src/`。对外契约是一个函数：

```python
def solve_task(context: TaskContext, env: DockerEnv, model: ModelClient) -> str:
    """Pipeline: localize → generate → validate → refine → select best patch."""
    ...
    return best_patch  # unified diff string
```

| 模块 | 职责 |
|------|------|
| `src/agent.py` | `solve_task` 编排 |
| `src/localize.py` | 分层定位 |
| `src/generate.py` | 规划 + 候选生成 |
| `src/validate.py` | 测试执行与排序 |
| `src/refine.py` | 精炼循环 |
| `src/config.py` | `AgentConfig`（候选数、预算、验证模式） |
| `utils/*` | 框架：Docker、模型、I/O、任务加载 |

```text
.
├── main.py              # CLI：生成 predictions.jsonl
├── evaluate.py          # CLI：官方 SWE-bench 评估
├── src/                 # ★ SWE-SolveAgent（solve_task 流水线）
├── utils/               # Docker / 模型 / 补丁工具
├── scripts/             # 数据集下载与镜像拉取
├── tests/               # 各阶段单元测试
├── assets/
│   └── pipeline.jpg     # 架构图
├── requirements.txt
├── swebench_tasks.txt   # instance_id 列表
└── .env.example
```

---

## 快速开始

### 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置模型

复制 `.env.example` → `.env` 并填写 API：

```env
apikey=your-api-key
base=https://api.example.com/v1
model=deepseek-v3.2
```

### 3. 准备任务与数据

`swebench_tasks.txt` — 每行一个 `instance_id`：

```text
# one instance per line
astropy__astropy-12907
django__django-11099
```

```bash
# 本地下载 SWE-bench Lite
python scripts/download_dataset.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir data/princeton-nlp__SWE-bench_Lite

# 拉取任务对应 Docker 镜像（体积大，建议先拉少量）
python -m scripts.pull_images --tasks swebench_tasks.txt
```

### 4. 生成预测

```bash
python main.py \
  --tasks swebench_tasks.txt \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --output predictions.jsonl \
  --run-id demo
```

### 5. 评估

```bash
python evaluate.py \
  --predictions predictions.jsonl \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --run-id demo \
  --max-workers 1
```

---

## 核心 API

```python
from src.agent import solve_task

patch: str = solve_task(context, env, model)
# patch 作为 model_patch 写入 predictions.jsonl
```

每行预测格式：

```json
{"instance_id": "...", "model_name_or_path": "...", "model_patch": "diff --git ..."}
```

---

## 技术栈

- **Python 3.10+**
- **SWE-bench** harness + Docker 沙箱
- **LLM**（OpenAI 兼容 HTTP API）
- **pytest** 单元测试

---

## 参考

- [SWE-bench](https://www.swebench.com/) — 真实 GitHub Issue 修复基准
- [SWE-bench harness 文档](https://www.swebench.com/SWE-bench/reference/harness/)
- PatchPilot / Agentless 风格多阶段修复智能体（localize → generate → validate → refine）

---

## 许可

课程 / 作品集项目。SWE-bench 为第三方软件，请遵循其 harness 许可证。
