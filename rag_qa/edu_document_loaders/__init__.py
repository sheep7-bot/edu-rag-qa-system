# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/__init__.py —— 加载器包入口
================================================================

【这个文件干什么的？】
   把同目录下的 4 个加载器"汇总导出"，外部只需：
       from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, ...
   而不用记住每个类在哪个文件里。

【Python 包知识：__init__.py】
   文件夹里有 __init__.py 才会被 Python 当成"包"。
   在 __init__.py 里写 import，等价于"把子模块的东西搬到包门口"。

【为什么用 try/except ImportError 包着？】
   每个加载器依赖不同的第三方库（PyMuPDF、python-docx、python-pptx...）。
   如果某个库没装，只跳过那一个加载器的导入，
   而不是让整个包导入失败（这叫"优雅降级"）。
"""
import sys
import os
# 把上一级目录（rag_qa）加进 sys.path，保证包外直接运行时也能找到 base 包
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入配置（部分调试入口会用到）
from base.config import config

# 添加配置中的路径到 sys.path（原作者的预留代码，当前未使用）
# sys.path.append(config.EDU_DOCUMENT_LOADERS_DIR)

# 条件导入各个文档加载器，允许在缺少依赖时继续运行
try:
    from .edu_docloader import *      # Word 加载器
except ImportError:
    pass

try:
    from .edu_pptloader import *      # PPT 加载器
except ImportError:
    pass

try:
    from .edu_imgloader import *      # 图片加载器
except ImportError:
    pass

try:
    from .edu_pdfloader import *      # PDF 加载器
except ImportError:
    pass
