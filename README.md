# EduRAG 智能问答系统

面向教育场景的双层智能问答系统：**FAQ 快通道（BM25）+ RAG 生成通道**。支持多格式文档解析入库、混合检索与重排序、网页端流式多轮对话，并内置 RAGAS 评估流水线。

## 功能特性

- **双层问答架构**：高频问题走 FAQ 快通道（BM25 + 置信阈值），命中后无需调用大模型即可返回；长尾问题走 RAG 生成通道
- **多格式文档入库**：PDF / Word / PPT / 图片 OCR 解析，父子分块（父 1200 字 / 子 300 字）
- **混合检索与重排序**：BGE-M3 稠密 + 稀疏向量混合检索，加权融合后经 BGE-reranker 精排，命中子块回溯完整父段上下文
- **查询策略路由**：BERT 问题分类（通用 / 专业）；支持直接检索、子问题拆分、HyDE 假设答案三种查询方式
- **工程能力**：FastAPI + WebSocket 逐字流式输出；Redis 缓存；MySQL 会话记忆（最近 5 轮）
- **效果评估**：RAGAS 四维指标（忠实度 / 答案相关性 / 上下文精度 / 上下文召回）

## 系统架构

```
用户提问
  ├─ FAQ 快通道：Redis 精确匹配 → BM25 模糊匹配（阈值 0.85）→ 命中直接返回
  └─ RAG 通道：BERT 分类路由 → 查询改写（直接 / 子查询 / HyDE）
               → Milvus 稠密+稀疏混合检索 → 加权融合 → BGE-reranker 精排
               → 父块上下文 + 会话历史 → 大模型流式生成
```

## 技术栈

Python · FastAPI · WebSocket · LangChain · BGE-M3 · Milvus · BERT · Redis · MySQL · DeepSeek · RAGAS · RapidOCR

## 快速开始

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

2. 启动依赖服务：MySQL、Redis、Milvus（docker compose，端口 19530）

3. 初始化数据库：导入 `subjects_kg.sql`

4. 配置：
   - 环境变量 `DEEPSEEK_API_KEY`（推荐），或在 `config.ini` 的 `[llm]` 中填写
   - 本地模型路径：`rag_qa/core/vector_store.py`（BGE-M3、BGE-reranker）与 `rag_qa/core/query_classifier.py`（BERT 分类器）

5. 离线建库（将文档放入 `rag_qa/data/<学科>_data/` 后执行）：

   ```bash
   python -m rag_qa.core.document_processor   # 加载与父子分块
   python -m rag_qa.core.vector_store         # 向量化并写入 Milvus
   ```

6. 启动 Web 服务：

   ```bash
   python app.py   # http://localhost:8080
   ```

## 目录结构

```
├── app.py                        # FastAPI 入口（REST + WebSocket）
├── config.ini                    # 配置文件
├── subjects_kg.sql               # FAQ 题库与会话历史建库脚本
├── base/                         # 配置与日志
├── mysql_qa/                     # FAQ 快通道（BM25 + Redis + MySQL）
├── rag_qa/
│   ├── core/                     # 文档处理、向量检索、策略选择、RAG 主流程
│   ├── edu_document_loaders/     # 多格式加载器（PDF / Word / PPT / 图片 OCR）
│   ├── edu_text_spliter/         # 中文递归切分器
│   └── rag_assesment/            # RAGAS 评估脚本与样例集
└── static/                       # 前端页面
```

## 效果评估

`rag_qa/rag_assesment/` 内置 RAGAS 评估脚本与样例数据集，四维指标明细见 `ragas_evaluation_results.csv`（小样本样例）。
