# -*- coding: utf-8 -*-
"""
rag_qa/core/rag_system.py —— RAG 智能问答主系统（把各模块串起来）
================================================================

【这个文件的核心地位】
   它是 RAG 链路的"总指挥"，把前面所有模块组装成完整流程：

   用户问题
     │
     ├─ 1. BERT 分类（query_classifier）──── 通用知识 → 不检索，直接让 LLM 答
     │                                   └─── 专业咨询 ↓
     ├─ 2. 策略检索（strategy_selector）→ 向量库召回相关父块
     ├─ 3. 提示词拼装（prompts）→ 上下文+历史+问题 → 完整 Prompt
     └─ 4. LLM 流式生成（llm.stream）→ 逐字吐出答案

【重要：generate_answer 是一个"生成器"（generator）】
   函数里用了 yield 关键字 → 调用它不会立刻执行，而是返回生成器对象。
   外面用 for token in self.rag.generate_answer(...) 一个个取。
   好处：可以边生成边显示（打字机效果），用户不用等完整答案。
   faq_milvus.py 和 app.py 的 WebSocket 就靠这个特性做流式输出。

【init_chat_model 是什么？】
   LangChain 提供的"通用模型初始化器"：
   传模型名即可自动选择对应厂商 SDK（openai/deepseek/anthropic...），
   本项目用 DeepSeek（OpenAI 兼容协议），配置见 config.ini 的 [llm] 节。
"""
from langchain.chat_models import init_chat_model
from base.config import config
from base.logger import logger
from rag_qa.core.prompts import RAGPrompts              # 提示词模板
from rag_qa.core.query_classifier import QueryClassifier  # BERT 问题分类器
from rag_qa.core.vector_store import VectorStore        # 向量检索引擎
from rag_qa.core.strategy_selector import StrategySelector  # 检索策略选择器


class RAGSystem:
    """RAG 问答系统核心类（装配 + 生成答案）。"""

    def __init__(self):
        logger.info("初始化 RAG 智能问答系统...")

        # 1. 初始化问题分类器（BERT 微调模型）
        self.classifier = QueryClassifier()
        logger.info("问题分类器加载完成")

        # 2. 初始化向量存储（Milvus + BGE-M3 + Reranker）
        self.vector_store = VectorStore()
        logger.info("向量检索引擎加载完成")

        # 3. 初始化大模型
        self.llm = init_chat_model(
            model=config.LLM_MODEL,
            temperature=0.1  # 低温度，保证答案专业、稳定、少发挥
        )
        logger.info("大模型初始化完成")


    def generate_answer(self, query, source_filter=None, history=None):
        """
        根据用户问题生成最终答案（RAG 核心入口，生成器函数）。

        :param query:         用户问题
        :param source_filter: 学科过滤（可选，如 "ai"）
        :param history:       最近几轮对话历史 [{"question":..., "answer":...}, ...]
        :yield:               答案文本片段（流式）
        """
        # 空问题保护
        if not query or len(query.strip()) == 0:
            return "请输入有效的问题~"

        # 获取历史记录，拼接为字符串（格式："Q:... A:..." 每行一轮）
        if history is None:
            history = []
        history_context = "\n".join([f"Q:{i['question']}\nA:{i['answer']}" for i in history])

        # ====================== 1. 问题分类 ======================
        # BERT 判断：通用知识 or 专业咨询
        query_category = self.classifier.predict_category(query.strip())
        logger.info(f"问题分类结果: [{query_category}] → {query}")

        # ====================== 2. 构建提示词 ======================
        if query_category == "通用知识":
            # 通用知识：不用知识库，context 占位为"通用知识"，直接回答
            prompt = RAGPrompts.rag_prompt().format(
                context="通用知识",
                history=history_context,
                question=query,
                phone=config.CUSTOMER_SERVICE_PHONE
            )
        else:
            # 专业咨询：从 Milvus 混合检索 + 重排序拿上下文
            logger.info("开始执行向量检索...")
            # 下面这行是"直接检索"的旧写法，已被策略选择器取代（保留供对比学习）：
            # documents = self.vector_store.hybrid_search_with_reranker(query=query, source_filter=source_filter)
            ss = StrategySelector()  # 新建策略选择器（内部会初始化检索资源）
            documents = ss.strategy_select(query=query, source_filter=source_filter)

            if documents:
                # 拼接父块上下文（高质量长文本），块与块之间空两行分隔
                context = "\n\n".join([doc.page_content for doc in documents])
                logger.info(f"检索到 {len(documents)} 条有效参考内容")
            else:
                # 检索为空：给模型一个明确提示，它会引导用户联系客服
                logger.warning("向量库未检索到相关内容")
                context = "未找到相关专业知识"

            # 构建 RAG 提示词（检索上下文 + 历史 + 问题 + 电话）
            prompt = RAGPrompts.rag_prompt().format(
                context=context,
                history=history_context,
                question=query,
                phone=config.CUSTOMER_SERVICE_PHONE
            )

        # 调试日志：只打前 300 字，避免日志被超长提示词刷屏
        logger.debug(f"最终发送大模型提示词: {prompt[:300]}...")

        # ====================== 3. 调用大模型生成答案（流式） ======================
        try:
            # llm.stream(prompt)：流式接口，返回一个迭代器，
            # 每轮拿到一小段输出 chunk（chunk.content 是文本片段）
            for chunk in self.llm.stream(prompt):
                if chunk.content:      # 有的 chunk 只有元数据没有文字，跳过
                    yield chunk.content  # 把文字片段"吐"给调用方

        except Exception as e:
            # LLM 调用失败：兜底话术，不让接口 500
            logger.error(f"LLM 流式错误: {e}")
            yield f"抱歉，处理出错，请联系客服：{config.CUSTOMER_SERVICE_PHONE}"


# ============================================================
# 【单独调试】python -m rag_qa.core.rag_system
# 注意：generate_answer 是生成器，要用 "".join(...) 把片段拼起来。
# 下面三种问法分别会触发：直接检索 / 子查询检索 / 假设问题检索。
# ============================================================
if __name__ == '__main__':
    # 测试 RAG 主系统
    rag_system = RAGSystem()

    # 测试三种策略都会自动走
    #answer = rag_system.generate_answer(query="AI智能应用学科的课程目录是怎样的")  # 直接检索
    #answer = rag_system.generate_answer(query="人工智能学科和AI智能应用学科区别是什么")  # 子查询检索
    answer = "".join(rag_system.generate_answer(query="人工智能学科在教育领域的应用有哪些？"))  # 假设问题检索
    print(answer)
