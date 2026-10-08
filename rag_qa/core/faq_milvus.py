# -*- coding: utf-8 -*-
"""
rag_qa/core/faq_milvus.py —— 双层集成问答系统（FAQ 快通道 + RAG 慢通道）
================================================================

【这个文件是整条问答链路的"总入口"】
   app.py（Web 后端）和 rag_qa/main.py（命令行）调用的都是这里的 IntegratedQASystem。

【双层架构（本项目的招牌设计）】
   用户问题
     │
     ▼
   ┌─ 第 1 层：BM25 FAQ 快通道（BM25Search）────────────────┐
   │  在 MySQL/Redis 题库里做关键词匹配 + 阈值判断            │
   │   命中（常见问题）→ 直接返回答案，0 成本、毫秒级          │
   │   未命中         → need_rag=True，进入第 2 层           │
   └────────────────────────────────────────────────────┘
     │
     ▼
   ┌─ 第 2 层：RAG 慢通道（RAGSystem）─────────────────────┐
   │  BERT 分类 → 策略检索 → 大模型生成                      │
   └────────────────────────────────────────────────────┘
   这样"高频问题走缓存、长尾问题走大模型"，兼顾速度与覆盖率。

【为什么 query() 是生成器？】
   方法里用了 yield，每次产出一个 (文本片段, 是否结束) 二元组：
       ("你", False) → ("好", False) → ... → ("", True)
   调用方（app.py / 命令行）拿到片段就能立刻显示，实现"打字机"流式效果。
   FAQ 通道没有原生流式，这里用"逐字符 + sleep(0.01)"模拟同样体验；
   RAG 通道则直接透传大模型的真流式 token。
"""
import time  # sleep 用来模拟打字机节奏

from base.logger import logger
from mysql_qa.retrieval.bm25_search import BM25Search  # FAQ 快通道
from rag_qa.core.rag_system import RAGSystem           # RAG 慢通道

from mysql_qa.db.mysql_client import MySQLClient       # 会话历史存取


class IntegratedQASystem:
    """集成问答系统：一碗水端平 FAQ 与 RAG 两条通道。"""

    def __init__(self):
        # 三个核心组件一次初始化，后续 query 复用
        self.bm25 = BM25Search()            # FAQ 检索（连 Redis/MySQL）
        self.rag = RAGSystem()              # RAG 系统（连 Milvus + 大模型）
        self.mysql_client = MySQLClient()   # 会话历史（MySQL）
        logger.info("集成问答系统初始化完成")

    def query(self, q, source_filter=None, session_id=None):
        """
        问答统一入口（生成器）。

        :param q:             用户问题
        :param source_filter: 学科过滤（如 "ai"，None=不限）
        :param session_id:    会话编号（用于存/取多轮历史）
        :yield:               (文本片段, 是否结束)
        """
        # ========== 第 1 层：FAQ 快通道 ==========
        # threshold=0.85：置信度足够高才认 FAQ 命中，避免错答
        faq_ans, need_rag = self.bm25.search(q, 0.85)

        if faq_ans:
            # ---- FAQ 命中：本轮对话存入 MySQL（会话记忆）----
            if session_id:
                self.mysql_client.update_session_history(session_id, q, faq_ans)
            # 逐字符流式推送答案，模拟打字效果
            for char in faq_ans:
                yield char, False
                time.sleep(0.01)  # 每个字间隔 10ms，让前端看到"打字"
            yield "", True  # 空字符串+True = 流式传输结束信号
            return

        # ========== 第 2 层：RAG 慢通道 ==========
        if need_rag:
            logger.info("FAQ无匹配，切换RAG检索")

            # 查历史：多轮对话记忆（最近 5 轮，来自 MySQL）
            history = self.mysql_client.fetch_recent_history(session_id) if session_id else []

            rag_answer = ""  # 累积完整答案（流完后要存库）
            # 调用 RAG 流式生成：每个 token 透传给调用方
            for token in self.rag.generate_answer(q, source_filter, history):
                rag_answer += token
                yield token, False

            # ---- RAG 回答完毕：把"问题+完整答案"存入 MySQL ----
            if session_id:
                self.mysql_client.update_session_history(session_id, q, rag_answer)
            yield "", True
            return

        # ========== 兜底：两通道都没有结果 ==========
        no_ans = "未找到答案，请联系人工客服。"
        for char in no_ans:
            yield char, False
            time.sleep(0.02)  # 兜底文案放慢一点
        yield "", True


# ============================================================
# 【单独调试】python -m rag_qa.core.faq_milvus
# 命令行问答：FAQ 命中秒回；未命中自动走 RAG。
# ============================================================
if __name__ == "__main__":
    print("===== EduRAG集成问答 | exit退出 =====")
    qa = IntegratedQASystem()

    while True:
        q = input("请输入问题：").strip()
        if q.lower() == "exit":
            logger.info("用户退出")
            break
        subj = input("学科过滤(回车跳过)：").strip() or None
        print("\n🚀 正在处理...\n")
        print("-" * 60)
        print(f"✅ 回答：")
        # 注意：生成器每次只能遍历一次；这里固定 session_id=123 演示多轮记忆
        gen = qa.query(q, subj, 123)
        for chunk, finished in gen:
            print(chunk, end="", flush=True)  # end="" 不换行；flush 立即输出
            if finished:
                break
        print("-" * 60, "\n")
