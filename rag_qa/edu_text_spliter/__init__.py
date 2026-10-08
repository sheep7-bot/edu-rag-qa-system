# -*- coding: utf-8 -*-
"""
rag_qa/edu_text_spliter/__init__.py —— 文本分割器包入口
================================================================

【作用】
   汇总导出本包的两个分割器，外部可以这样导入：
       from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter, AliTextSplitter

【条件导入】
   AliTextSplitter 依赖 modelscope 等重依赖，没装时只跳过它，
   保证 ChineseRecursiveTextSplitter（主链路用的）不受影响。
"""
# 星号导出：把中文分割器模块里的公共名字都引进来
from .edu_chinese_recursive_text_splitter import *

# 条件导入需要额外依赖的模块（modelscope 等）
try:
    from .edu_model_text_spliter import *
except ImportError:
    pass
