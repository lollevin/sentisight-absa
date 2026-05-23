# SentiSight（慧感）— 技术文档

## 1. 项目概述

SentiSight 是一个基于大语言模型（LLM）的 Aspect-Based Sentiment Analysis（ABSA）营销洞察引擎。它能够从用户评论中自动提取"方面"级别的情感倾向——不只是判断好评还是差评，而是告诉你"配送很快（正面，97%置信度）但包装很差（负面，95%置信度）"——并生成可操作的营销建议。

传统工具（如 VADER）只能输出一个笼统的整体情感分数，无法帮助营销人员定位具体需要改进的产品维度。SentiSight 通过 LLM 驱动的 ABSA 解决了这个问题。

---

## 2. AI 技术

### 2.1 核心技术：LLM-based Aspect-Based Sentiment Analysis

ABSA 的传统做法需要训练专用 NLP 模型（如 BERT-based 分类器），但这类模型需要大量标注数据，且泛化能力有限。SentiSight 采用 LLM 零样本（zero-shot）方案：

- 通过精心设计的 system prompt 引导 LLM 扮演营销分析师角色
- 要求 LLM 以严格 JSON 格式输出：方面名称、情感极性、置信度、原文引用、关键词、营销建议
- 无需训练数据即可适用于任何产品领域

### 2.2 Prompt Engineering 策略

系统使用了一个结构化的 system prompt，关键设计要素：

1. **角色设定**："You are an expert marketing analyst specializing in ABSA"
2. **输出格式约束**：提供完整的 JSON schema 示例，要求只返回 JSON
3. **质量规则**：
   - 只识别评论中实际提及的方面
   - 置信度必须在 0.0-1.0 之间
   - 营销建议必须具体可操作（不能笼统）
   - 若无法识别明确方面，使用 "overall" 作为方面名
4. **Temperature 设置**：0.3，确保输出一致性和可复现性

### 2.3 支持的 LLM Provider

| Provider | 模型 | 特点 |
|----------|------|------|
| DeepSeek | deepseek-chat, deepseek-reasoner | 推荐，费用极低，中文能力强 |
| OpenAI | gpt-4o-mini, gpt-4o, gpt-3.5-turbo | 最广泛使用 |
| Groq | llama-3.3-70b, llama-3.1-8b, gemma2-9b | 免费层，推理速度最快 |
| Gemini | gemini-2.0-flash, gemini-1.5-flash/pro | Google 免费层 |

所有 provider 通过 OpenAI-compatible API 统一调用，使用 openai Python 库。对于不支持 `response_format=json_object` 的 provider（Groq、Gemini），系统能自动处理 markdown code fence 包裹的 JSON。

### 2.4 传统基线：VADER

系统集成了 VADER（Valence Aware Dictionary and sEntiment Reasoner）作为传统情感分析的基线对照。VADER 是基于规则+词典的方法，只能输出一个 -1 到 +1 的 compound score 和整体标签，无法识别具体方面。这正好展示了 LLM-based ABSA 相比传统方法的优势。

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ │
│  │ Tab 1    │ │ Tab 2    │ │ Tab 3  │ │ Tab 4    │ │
│  │ 单条分析 │ │ 批量分析 │ │ 对比   │ │ 测试验证 │ │
│  └────┬─────┘ └────┬─────┘ └───┬────┘ └────┬─────┘ │
├───────┼────────────┼───────────┼───────────┼───────┤
│       ▼            ▼           ▼           ▼        │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │absa_    │ │tradition │ │visualizer│ │evalua  │  │
│  │engine.py│ │al.py     │ │.py       │ │tor.py  │  │
│  │LLM调用  │ │VADER分析 │ │Plotly图表│ │评估引擎│  │
│  └────┬────┘ └──────────┘ └──────────┘ └───┬────┘  │
│       │                                     │       │
├───────┼─────────────────────────────────────┼───────┤
│       ▼                                     ▼       │
│  ┌─────────┐                          ┌──────────┐  │
│  │DeepSeek │  OpenAI  Groq  Gemini    │labeled_  │  │
│  │  API    │                          │test_data │  │
│  └─────────┘                          │.csv (30条)│  │
│                                       └──────────┘  │
│  ┌──────────────┐                                   │
│  │lang.py       │  中英双语翻译 (EN/ZH)             │
│  └──────────────┘                                   │
└─────────────────────────────────────────────────────┘
```

核心模块说明：

| 模块 | 行数 | 职责 |
|------|------|------|
| `app.py` | ~930 | Streamlit UI，4 个 Tab，CSS 样式，主题切换 |
| `absa_engine.py` | 126 | LLM API 调用，prompt 管理，4 个 provider 适配 |
| `traditional.py` | 35 | VADER 情感分析（含缓存） |
| `evaluator.py` | 165 | 评估引擎：Precision/Recall/F1，ABSA vs VADER 对比 |
| `visualizer.py` | 177 | Plotly 图表：柱状图、饼图、词云、仪表盘 |
| `lang.py` | 322 | 中英双语完整翻译字典 |

---

## 4. 实现细节

### 4.1 API 调用流程

1. 用户选择 provider + model + 输入 API key
2. `get_client()` 根据 provider 创建对应的 OpenAI client（不同 base_url）
3. `analyze_review()` 发送 system prompt + 用户评论，temperature=0.3
4. 对支持 JSON mode 的 provider 设置 `response_format={"type": "json_object"}`
5. 解析返回的 JSON，处理 markdown code fence 包裹情况
6. 标准化字段（aspects, overall_sentiment, keywords, marketing_suggestions）避免 KeyError

### 4.2 批量分析

`analyze_batch()` 逐条处理评论，每条独立调用 API。通过 progress_callback 实时更新 Streamlit 进度条。失败的评论返回 error 标记而非中断整个批次。

### 4.3 评估引擎

`evaluator.py` 实现了完整的评估流程：
- 加载人工标注数据（review + aspect + expected_sentiment）
- 运行 ABSA 模型获取预测结果
- 方面匹配：通过 exact/substring matching 将预测方面与标注方面对齐
- 计算：Detection Rate、Sentiment Accuracy、Per-class Precision/Recall/F1、Macro F1
- VADER 基线：将 VADER 的整体标签与每个方面的标注对比（有意展示 VADER 的局限性）

### 4.4 性能优化

- `@st.cache_data` 缓存 `load_labeled_data()` — CSV 只读一次
- `@st.cache_data` 缓存 `analyze_vader()` — 相同输入复用结果
- Demo VADER 结果一次计算后缓存
- 主题 CSS 动态生成，避免重复 DOM 操作

### 4.5 用户体验

- 深色/浅色主题切换（config.toml 设浅色为基础，深色用 CSS 覆盖）
- 中英双语（lang.py 完整翻译字典，322 行）
- `st.metric()` 展示 KPI + delta 对比值
- 错误处理：try-except 包裹所有 API 调用，显示中文友好提示
- 响应式布局：st.columns() 实现多列展示

---

## 5. 部署方案

### 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud 部署
- 项目已配置 `.streamlit/config.toml`
- 添加 API key 到 Streamlit Secrets 或 `.env`
- GitHub push 后自动 redeploy

---

## 6. 局限性与未来改进

1. **样本量**：测试数据集仅 30 条，未来可扩展到更大规模
2. **方面覆盖**：LLM 零样本方案可能遗漏罕见方面，可考虑 few-shot 示例
3. **成本**：批量分析每条评论都调用一次 API，大批量时成本较高。可考虑:
   - 多评论合并到一次 API 调用
   - 使用更便宜的模型做预筛选
4. **实时性**：当前是逐条处理的同步架构，未来可改为异步并发
