# devknow-py · 研发知识问答系统

面向研发场景的 RAG 问答服务：语料是 Spring Boot 官方文档与 GitHub Issue / PR（**全英文**），用户用**中文**提问，系统跨语言检索英文原文，并用中文作答，答案强制附带原文链接。

页面和接口同源，一条命令起服务。

## 效果

| 指标 | 数值 | 说明 |
|---|---|---|
| 知识库规模 | 5,302 篇文档 / 8,715 个 chunk | Issue 4,207 / PR 875 / 文档 220 |
| 向量覆盖率 | 100% | bge-m3，1024 维 |
| Recall@5 | **0.92** | 50 条标准题集，窗口 20，document 级 |
| MRR@5 | 0.82 | 同上 |
| Hit@1 | 0.76 | 同上 |
| 无关问题拒答 | 5/5 | 空结果即拒答依据 |

同题集下，双路召回 + 精排（0.92）明显优于单路向量召回（0.72）。

## 链路

```
GitHub ─┐
        ├─► 采集 / 解析 / 按标题层级切分 ─► PostgreSQL + pgvector
文档 ───┘                                      │
                                               │
提问 ──► 向量召回 ─┐                           │
                     ├─► RRF 融合 ─► 精排 ─► 取 Top-5 ─► LangGraph 生成 ─► SSE 流式返回
提问 ──► 全文召回 ─┘   （只看名次）  （bge-reranker）        （附原文引用）
```

三个关键设计：

1. **RRF 融合只看名次，不看分数**。余弦相似度（0~1）和 `ts_rank`（0~0.1）不同量纲，直接加权求和等于偷偷给向量路加权重。
2. **按 `document_id` 归并**。同一文档的多个 chunk 只保留最高分那个，否则 Top-5 会退化成“一篇文档的 5 个片段”。
3. **拒答依据跟着精排开关走**。精排开启时结果已按阈值过滤，空列表才代表“库中无相关内容”。

## 目录

| 目录 | 内容 |
|---|---|
| `app/ingest/` | 采集、解析、切分、向量化（`python -m app.ingest`） |
| `app/retrieval/` | 双路召回、RRF 融合、精排 |
| `app/agent/` | LangGraph 状态图：规划 → 工具调用 → 生成 |
| `app/conversation/` | 会话读写、上下文裁剪、摘要压缩 |
| `app/api/` | HTTP + SSE 接口 |
| `static/` | 演示页（与接口同源） |
| `eval/` | 评测题集 |
| `scripts/` | 评测与自检脚本 |
| `tests/` | 单测 |

## 快速开始

```powershell
# 1. 依赖
pip install -e .

# 2. 配置（照 .env.example 建 .env，填 SILICONFLOW_API_KEY 与数据库连接）
copy .env.example .env

# 3. 确认库里有数据
python -m app.ingest stats

# 4. 起服务（页面与接口同时可用）
python -m app
```

打开 <http://127.0.0.1:8000/> 。

> 启动约需 20 秒（导入 langgraph 较慢），看到 `Uvicorn running on ...` 再开浏览器。

环境：Python 3.10、PostgreSQL + pgvector。模型走 OpenAI 兼容协议：`deepseek-ai/DeepSeek-V3`、`BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3`。

## 数据同步

```powershell
python -m app.ingest sync-issues --source 1 --since 2025-01-01
python -m app.ingest sync-docs   --source 2 --ref main
python -m app.ingest embed       --limit 1000
python -m app.ingest stats
```

采集是增量且幂等的：按 `(source_id, doc_type, external_id, version)` 定位文档，正文 hash 未变则跳过切分。`GITHUB_TOKEN` 不填也能跑，但匿名限流 60 次/小时，全量同步会卡住。

## 评测

```powershell
python -m scripts.eval_java_set     # 标准题集，口径对齐（Recall@5 / MRR@5 / Hit@1）
python -m scripts.eval_retrieval    # 自建 50 条中文长问句，更难，另跑一组消融
```

两套题集的用途不同，数字不可混用：

- `eval/java-eval.json`：known-item 占 60%，用于**跨实现对比**
- `scripts/eval_retrieval.py` 内置的 50 条：全中文长问句，用于**衡量真实提问难度**（Recall@5 0.58）

## 已知限制

- **没有中文分词**。全文路只吃 ASCII 词（`extract_english_terms`），纯中文提问实际只有向量路在工作。
- **缺中文术语映射**。中文“打包成可执行 jar”检索不到英文 `executable archive`，这条在两套题集里都失败。
- **精排对官方文档类有小幅负作用**（Recall@5 0.5714 → 0.5429），对 Issue/PR 类是大幅正作用（0.2667 → 0.6667）。推测与送审文本截断 800 字符有关，尚未验证。
- **无鉴权、无限流**。学习项目定位，生产环境不可直接使用。
