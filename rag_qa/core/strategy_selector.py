# -*- coding: utf-8 -*-
"""
rag_qa/core/strategy_selector.py —— RAG 检索策略选择器（高级 RAG 技巧）
================================================================

【这个文件解决什么问题？】
   用户的问题"千奇百怪"，只用一种检索方式效果不稳定：
     - 简单问题（"课程大纲有哪些"）→ 直接检索就够了；
     - 复杂问题（"AI学科和Java学科的区别、各自就业方向"）→ 一句话里两个主题，
       整句做向量检索会"语义稀释"，拆成子问题分别查更准；
     - 口语化问题（"那个学费贵不贵来着"）→ 关键词少、语义模糊，
       先让大模型生成一段"假设答案"，用答案去检索（HyDE 技巧）更准。
   本文件让大模型当"调度员"，自动挑最合适的策略。

【三种策略总结】
   1. 直接检索     ：hybrid_search_with_reranker(query)
   2. 子查询检索   ：LLM 拆问题 → 逐个检索 → 合并去重
   3. HyDE 假设问题：LLM 生成假设答案 → 用假设答案检索

【工程细节：兜底思维】
   每个策略都有 try/except：LLM 挂了或返回空 → 返回空列表/退化到直接检索，
   绝不让整个问答流程崩掉。写生产代码要处处有这个意识。
"""
from base.logger import logger
from langchain.chat_models import init_chat_model  # 统一的"初始化聊天模型"入口

from base.config import config as conf
from rag_qa.core.prompts import RAGPrompts          # 提示词模板（见 prompts.py）
from rag_qa.core.vector_store import VectorStore    # 向量检索引擎（见 vector_store.py）


class StrategySelector:
    """智能检索策略选择器。"""

    def __init__(self):
        # 1. 初始化大模型：temperature=0.3 略大于 0，兼顾稳定与灵活
        self.llm = init_chat_model(model=conf.LLM_MODEL, temperature=0.3)

        # 2. 初始化向量存储（内部会加载 BGE-M3、连接 Milvus、加载重排序模型）
        self.vector_store = VectorStore()
        logger.info("检索策略引擎初始化完成")

    def strategy_select(self, query, source_filter=None):
        """
        核心入口：根据用户问题自动选择最优检索策略。

        :param query:         用户原始问题
        :param source_filter: 学科过滤（None=全库）
        :return: 排序后的文档列表（父块）
        """
        # 空问题直接返回空，不浪费 LLM 调用
        if not query or not query.strip():
            logger.warning("输入问题为空，返回空上下文")
            return []

        # ====================== 1. LLM 自动选择检索策略 ======================
        # 用策略选择模板问 LLM，期望它只回答三个策略名之一
        strategy_prompt = RAGPrompts.strategy_select_prompt().format(query=query)
        strategy = self._call_llm(strategy_prompt).strip()
        logger.info(f"策略选择结果：【{strategy}】 → {query}")

        # ====================== 2. 执行对应检索策略 ======================
        ranked_docs = []
        if strategy == "子查询检索":
            ranked_docs = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == "假设问题检索":
            ranked_docs = self._retrieve_with_hyde(query, source_filter)
        else:
            # 兜底分支：模型答了别的/答歪了 → 按"直接检索"处理，最稳
            ranked_docs = self.vector_store.hybrid_search_with_reranker(
                query=query,
                source_filter=source_filter
            )

        # ====================== 3. 截取最终候选集 ======================
        final_docs = ranked_docs[:conf.CANDIDATE_M]  # 默认只留前 2 条，控制 token 成本
        logger.info(f"策略执行完成，最终返回 {len(final_docs)} 条上下文")
        return final_docs

    def _retrieve_with_hyde(self, query, source_filter):
        """
        HyDE 假设答案检索策略。
        流程：先生成假设答案 → 用"假设答案"的向量去检索 → 语义更匹配。
        """
        logger.info(f"执行 HyDE 假设答案检索：{query}")
        try:
            # 1) 让 LLM 生成一段"假设答案文档"
            hyde_prompt = RAGPrompts.hyde_prompt().format(query=query)
            hypo_answer = self._call_llm(hyde_prompt).strip()
            logger.info(f"HyDE 生成答案：{hypo_answer[:100]}...")  # 只打前100字方便看日志

            # 2) 关键区别：用生成的"假设答案"当检索词（而不是原始问题）
            return self.vector_store.hybrid_search_with_reranker(
                query=hypo_answer,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败：{str(e)}")
            return []  # 失败返回空，由上层走"未检索到内容"的兜底提示

    def _retrieve_with_subqueries(self, query, source_filter=None):
        """
        子查询拆分策略。
        流程：复杂问题 → LLM 拆成 N 个小问题 → 分别检索 → 按文本去重合并。
        """
        logger.info(f"执行 子查询拆分检索：{query}")
        try:
            # 1) 让 LLM 拆问题（模板要求每行一个子问题）
            subquery_prompt = RAGPrompts.subquery_prompt().format(query=query)
            subqueries_text = self._call_llm(subquery_prompt).strip()
            # 按换行切开，strip 去掉每行空白，过滤空行
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]

            if not subqueries:
                logger.warning("子查询生成为空，返回空列表")
                return []

            logger.info(f"拆分后子问题：{subqueries}")

            # 2) 逐个子问题检索，结果塞进同一个列表
            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_reranker(
                    query=sub_q,
                    source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子问题 [{sub_q}] 召回 {len(docs)} 条")

            # 3) 文本去重：
            #    用"内容 → 文档"的字典天然去重（相同内容只会留最后一次）
            unique_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_dict.values())

            logger.info(f"合并去重后：{len(unique_docs)} 条")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询策略失败：{str(e)}")
            return []

    def _call_llm(self, prompt):
        """
        统一 LLM 调用接口：简化格式、稳定返回、异常兜底。
        任何异常都返回 "直接检索"，保证上层永远有可用的策略名。
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM 调用异常：{str(e)}")
            return "直接检索"


# ============================================================
# 【单独调试】python -m rag_qa.core.strategy_selector
# 会先让 LLM 选策略再执行检索，观察日志里打印的选择结果。
# ============================================================
if __name__ == '__main__':
    ss = StrategySelector()
    test_query = "AI智能应用学科的课程体系和就业方向是什么？"
    docs = ss.strategy_select(test_query)
    logger.info(f"最终获取上下文数量：{len(docs)}")
