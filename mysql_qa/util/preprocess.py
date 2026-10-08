# -*- coding: utf-8 -*-
"""
mysql_qa/util/preprocess.py —— 中文文本预处理（分词）
================================================================

【这个文件在整个问答链路里的位置】
   FAQ 检索（BM25）之前的一小步，但非常关键：
   用户问题 → 【本文件：转小写 + 分词】→ 词列表 → 交给 BM25 算相似度

【为什么中文必须先分词？】
   英文单词之间天然有空格，按空格切开就行；
   中文是"连续字符串"，BM25 这类算法是按"词"匹配的，
   不切词就等于拿整句话去比对，效果极差。
   例如："AI课程学费" 需要切成 ["ai", "课程", "学费"]。

【工具：jieba（结巴分词）】
   最流行的 Python 中文分词库，安装：pip install jieba
   常用模式：
     jieba.lcut(text)          精确模式（本项目用的，适合文本分析）
     jieba.lcut_for_search(text) 搜索引擎模式，长词再切细
   例：jieba.lcut("我正在学习人工智能") → ['我', '正在', '学习', '人工智能']
"""
import jieba  # 中文分词库

from base.logger import logger  # 统一日志


def process_text(text):
    """
    文本预处理核心函数：统一小写 + jieba 中文分词。

    参数：
        text: 原始输入文本（如 "2026AI智能应用课程大纲"）
    返回：
        分词后的词汇列表（如 ['2026', 'ai', '智能', '应用', '课程', '大纲']）
        出错时返回空列表（保证上层不崩）。

    注意：.lower() 把英文统一转小写，
    这样 "AI" 和 "ai" 会被当作同一个词，提高匹配率。
    """
    logger.debug(f"开始文本预处理：{text}")

    try:
        # jieba.lcut = list cut，直接返回词列表（不是生成器）
        return jieba.lcut(text.lower())
    except (AttributeError, TypeError) as e:
        # 捕捉两类典型错误：
        #   AttributeError —— text 是 None（None 没有 .lower 方法）
        #   TypeError      —— text 不是字符串（如传入数字）
        logger.error(f"文本预处理异常: {e}")
        return []
