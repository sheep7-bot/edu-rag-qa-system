# -*- coding: utf-8 -*-
"""
rag_qa.core 包 —— RAG 核心模块集合
================================================================
阅读顺序建议（也是数据流顺序）：
   1. document_processor.py  文档加载 + 父子分块（离线建库）
   2. vector_store.py        向量化 + Milvus 混合检索 + 重排序
   3. query_classifier.py    BERT 问题分类（通用/专业）
   4. prompts.py             提示词模板
   5. strategy_selector.py   检索策略选择（直接/子查询/HyDE）
   6. rag_system.py          RAG 主流程（分类→检索→生成）
   7. faq_milvus.py          FAQ+RAG 双层总入口
"""
