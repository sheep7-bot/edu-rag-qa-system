# -*- coding: utf-8 -*-
"""
mysql_qa/retrieval/bm25_search.py —— BM25 关键词检索（FAQ 快速通道）
================================================================

【这个模块在整体架构里的位置】
   用户提问
      │
      ▼
   ┌───────────────────────────────────────────────┐
   │ 第一步：FAQ 快速通道（本文件）                    │
   │   1. Redis 精确匹配：问题一模一样 → 直接返回答案   │
   │   2. BM25 模糊匹配：算相似度，超过阈值 0.85 → 返回  │
   │   3. 都没命中 → 交给 RAG 大模型链路（need_rag=True）│
   └───────────────────────────────────────────────┘
   也就是说：能用现成 FAQ 秒回的，就不浪费大模型算力；
   这层"快慢双通道"设计是工业问答系统的常见做法。

【什么是 BM25？（信息检索的经典算法）】
   给定"一个查询"和"一堆文档"（本项目里每条 FAQ 问题=一篇文档），
   BM25 按三个思路打分：
     1. 词频（TF）    ：这个词在文档里出现越多，越相关（但会饱和，不会无限加分）；
     2. 逆文档频率(IDF)：越少见的词权重越高（"的"这种词几乎不加分）；
     3. 文档长度归一化 ：长文档天然占便宜，要除以长度做惩罚。
   本项目用 rank_bm25 库实现：BM25Okapi(分词后的文档列表)。

【为什么要 Softmax 归一化？】
   BM25 原始分数没有固定范围（可能是 0.3，也可能是 42），
   没法定统一阈值。Softmax 把一组分数压缩成总和为 1 的概率分布，
   最大分就能和 0.85 这种固定阈值比较了。
   实现要点：先减去最大值再取 exp（防止 exp 溢出成 inf）。

【架构：Redis 缓存 + MySQL 持久数据源（冷热分层）】
   启动时优先从 Redis 拿题库 → 拿不到再查 MySQL 并回填 Redis。
   答案也一样：先 Redis 后 MySQL，查到后同步缓存。
   这样热数据（高频问答）走内存，冷数据走磁盘，整体速度快几十倍。
"""
from rank_bm25 import BM25Okapi  # BM25 算法库：pip install rank-bm25

from mysql_qa.cache.redis_client import RedisClient   # Redis 缓存客户端
from mysql_qa.db.mysql_client import MySQLClient      # MySQL 数据源
from mysql_qa.util.preprocess import process_text     # 中文分词预处理
from base.logger import logger                        # 统一日志
import numpy as np                                    # 数值计算（Softmax 用）


class BM25Search:

    def __init__(self):
        """构造方法：建连接 → 加载题库 → 初始化 BM25 模型。"""
        # 创建 MySQL、Redis 客户端连接对象
        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient()
        # 声明 BM25 检索模型（先占位，_load 里赋值）
        self.bm25 = None
        # 原始问题列表（用于用下标反查问题原文和答案）
        self.original_questions = None
        # 分词后的问题列表（喂给 BM25 算相似度）
        self.tokenized_questions = None
        # 初始化 BM25 检索模型
        self._load()

    def _load(self):
        """私有的初始化 BM25 检索模型（下划线开头表示"内部方法，外部别调"）。"""
        # 1. 从 Redis 读取原始问题、分词问题
        original_key = "qa_original_questions"    # 原始问题缓存的键名
        tokenized_key = "qa_tokenized_questions"  # 分词问题缓存的键名
        self.original_questions = self.redis_client.get_data(original_key)
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 2. 如果缓存中没有，从 MySQL 中查询，并分词处理（缓存未命中 → 回源）
        if not self.original_questions or not self.tokenized_questions:
            # 查询 MySQL 原始问题
            self.original_questions = self.mysql_client.fetch_questions()
            # 对每条原始问题进行分词处理 → 得到二维列表 [[词,词],[词,词],...]
            self.tokenized_questions = [process_text(q) for q in self.original_questions]
            # 3. 将原始问题和分词问题回填进 Redis（下次启动就不查库了）
            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, self.tokenized_questions)

        # 4. 初始化 BM25 检索模型：把"文档库"（这里是全部分词后的问题）灌进去建索引
        self.bm25 = BM25Okapi(self.tokenized_questions)
        logger.info("BM25 检索模型初始化成功")

    def search(self, query, threshold=0.85):
        """
        BM25 检索查询方法。

        参数：
            query:     用户问题
            threshold: 归一化后的阈值（0~1）。分数 ≥ 阈值才认为"匹配成功"。
                       0.85 比较严格：宁可放过（去走 RAG），不可答错。
        返回：
            (answer: str, need_llm: bool) 二元组：
              - 命中：返回 (答案, False)   → 不需要大模型
              - 未命中：返回 (None, True)  → 需要切换 RAG 大模型链路
        """
        # 1. Redis 问题的精准匹配查询（相同问题多次提问 → 秒回）
        cache_answer = self.redis_client.get_answer(query)
        if cache_answer:
            return cache_answer, False

        # 2. 用户问题分词，交给 BM25 计算与题库中每个问题的相似度
        query_tokens = process_text(query)
        raw_scores = self.bm25.get_scores(query_tokens)  # 原始分值数组，长度=题库问题数

        # 归一化处理（把无界分变成 0~1 概率）
        softmax_scores = self._softmax(raw_scores)

        # 3. 查询最高分及其索引
        #    nanmax：忽略 NaN 取最大（题库里有空问题时会出 NaN）
        #    argmax：最大值的下标 → 对应题库里第几条问题
        best_score = np.nanmax(softmax_scores)
        best_idx = np.argmax(softmax_scores)

        # 4. 最高分与阈值对比
        if best_score < threshold:
            logger.info(f"FAQ匹配失败，置信阈值{best_score:.3f}")
            # 分数 < 阈值：无匹配答案，让上层启动 LLM/RAG
            return None, True
        else:
            # 分数 ≥ 阈值：匹配成功，拿到库里的原始问题文本
            original_question = self.original_questions[best_idx]
            # 先查缓存
            cache_answer = self.redis_client.get_answer(original_question)
            if cache_answer:
                logger.info(f"Redis缓存命中问题：{original_question}")
                answer = cache_answer
            else:
                # 缓存没有再查 MySQL
                answer = self.mysql_client.fetch_answer(original_question)
                logger.info(f"MySQL命中问题：{original_question}")
                # 同步到缓存（下次同样的匹配直接走缓存）
                self.redis_client.set_data(f"answer:{original_question}", answer)
            logger.info(f"FAQ匹配成功，置信阈值{best_score:.3f}")
            return answer, False

    def _softmax(self, scores):
        """
        私有归一化函数：将 BM25 无界原始分值转为 0~1 概率。

        数学公式：softmax(x_i) = e^(x_i) / Σ e^(x_j)
        数值稳定技巧：先整体减去最大值（结果不变但防止 e^x 溢出）。
        """
        temperature = 1  # 温度：越大分布越平滑，越小越尖锐（这里保持 1）
        scores = np.array(scores) / temperature
        exp_scores = np.exp(scores - np.max(scores))  # 减最大值防溢出
        return exp_scores / exp_scores.sum()          # 除以总和 → 总和为 1


# ============================================================
# 【单独调试】python -m mysql_qa.retrieval.bm25_search
# 先确保 MySQL、Redis 已启动且数据已导入。
# 测试句可改成题库里真实存在的问题，观察 FAQ 命中情况。
# ============================================================
if __name__ == '__main__':
    bm25 = BM25Search()
    answer, need_llm = bm25.search("2026AI智能应用课程大纲")
    if answer:
        print(f"FAQ答案：{answer}")
    if need_llm:
        print("FAQ没有结果，需要执行LLM的RAG")
