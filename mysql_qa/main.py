# -*- coding: utf-8 -*-
"""
mysql_qa/main.py —— FAQ 问答命令行入口（只测 FAQ 通道）
================================================================

【这个文件干什么的？】
   一个"最小可运行"的命令行测试入口：
   只包含 BM25(FAQ) 这条快通道，不接 RAG 大模型。
   用来单独验证：MySQL + Redis + BM25 这条链路通不通。

   完整版（FAQ + RAG 双层）在 rag_qa/core/faq_milvus.py，Web 入口是项目根的 app.py。

【跑法】
   1. 确认 MySQL、Redis 已启动，subjects_kg 数据已导入；
   2. 在项目根目录执行：python -m mysql_qa.main
   3. 输入问题回车，FAQ 命中就秒回；未命中提示"FAQ未找到答案"。
"""
import os
import sys

# 把项目根目录加入 sys.path：保证"直接运行本文件"时能找到 base、mysql_qa 等包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mysql_qa.retrieval.bm25_search import BM25Search


class MySQLQASystem:
    """简易 FAQ 问答系统：内部就一个 BM25 检索器。"""

    def __init__(self):
        # 初始化即加载 BM25 模型（内部会连 MySQL/Redis 并缓存题库）
        self.bm25 = BM25Search()

    def query(self, q):
        """查询入口：命中返回答案，未命中给提示。"""
        # threshold=0.85：相似度阈值，低于它视为没找到
        ans, need_llm = self.bm25.search(q, 0.85)
        if need_llm:
            # 这里没有接 RAG，只能提示；完整版会转去大模型链路
            ans = "FAQ未找到答案"
        return ans


if __name__ == "__main__":
    print("===== 问答系统启动 | exit退出 =====")
    qa = MySQLQASystem()
    # 循环问答，输入 exit 退出（命令行交互的经典写法）
    while True:
        q = input("请输入问题：").strip()
        if q.lower() == "exit":
            break
        print(f"✅ 答案：{qa.query(q)}\n")
