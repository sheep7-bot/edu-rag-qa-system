# -*- coding: utf-8 -*-
"""
rag_qa/core/document_processor.py —— 文档处理总管线（加载 → 父子分块）
================================================================

【这个文件在 RAG 数据准备阶段的位置】
   离线建库流水线的第一步：
      原始文档(各种格式) → ①加载器解析成文本 → ②父子分块 → ③子块向量化入 Milvus
   本文件负责 ① 和 ②；③ 在 vector_store.py 的 add_documents()。

【核心概念：父子分块（Parent-Child Chunking）—— 本项目的精华设计】
   为什么不直接把文档切一种块？因为"检索精度"和"上下文质量"是矛盾的：
     - 小块（子块 300 字）：语义聚焦，向量检索更准 → 用来"找"；
     - 大块（父块 1200 字）：上下文完整，大模型回答质量更高 → 用来"喂"。
   做法：
     1. 先把文档切成父块（大块）；
     2. 每个父块再切成多个子块（小块）；
     3. 子块存进 Milvus（带 parent_id、parent_content 两个字段）；
     4. 检索时用子块匹配，命中后返回它所属的"父块全文"给大模型。
   这样"检索用小块、生成用大块"，两头的好处都占了。

【技术：加载器映射表（工厂模式的雏形）】
   用字典把"文件后缀 → 加载器类"对应起来，
   以后要支持新格式（如 .csv），只在字典里加一行即可，不用改流程代码。

【跑法（建库第一步）】
   python -m rag_qa.core.document_processor
   它会加载 rag_qa/data 下所有文档、切好块并打印块数。
   第二步（向量化入库）在 vector_store.py 的 __main__ 里。
"""
import os
import sys
# 把项目根目录加入 sys.path（本文件位置：项目根/rag_qa/core/document_processor.py，往上三层）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime  # 生成入库时间戳

# LangChain 社区版文档加载器
from langchain_community.document_loaders import TextLoader            # .txt
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader  # .md

# 自定义中文文本分割器（见 edu_text_spliter/，原理见该包注释）
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter, AliTextSplitter
# LangChain 的 Markdown 分割器：按 # 标题层级切，更懂 Markdown 结构
from langchain_text_splitters import MarkdownTextSplitter
# 自定义 OCR 加载器（支持 PDF/Word/PPT/图片，见 edu_document_loaders/）
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader

# 项目基础配置与日志
from base.config import config as conf
from base.logger import logger


# ====================== 文件加载器映射表 ======================
# 支持的文件类型与对应加载器，统一管理便于扩展。
# 想支持新格式？在这里加一行 "后缀: 加载器类" 就行。
document_loaders = {
    ".txt": TextLoader,          # 纯文本（需指定 utf-8 编码）
    ".pdf": OCRPDFLoader,        # PDF（文字+大图 OCR）
    ".docx": OCRDOCLoader,       # Word（段落+表格+图片 OCR）
    ".ppt": OCRPPTLoader,        # 老版 PPT
    ".pptx": OCRPPTLoader,       # 新版 PPT
    ".jpg": OCRIMGLoader,        # JPG 图片
    ".png": OCRIMGLoader,        # PNG 图片
    ".md": UnstructuredMarkdownLoader  # Markdown
}


def load_documents_from_directory(directory_path):
    """
    从指定目录递归加载所有支持的文档文件。
    自动识别文件类型、添加元数据、异常捕获、日志记录。

    :param directory_path: 文档根目录路径（如 data，内含 ai_data、java_data 等子目录）
    :return: LangChain Document 列表

    【目录约定】
       data/
         ai_data/      → source 元数据 = "ai"
         java_data/    → source 元数据 = "java"
       "学科来源"靠目录名推断（去掉 _data 后缀），用于后续检索过滤。
    """
    documents = []  # 存储所有加载完成的文档
    supported_extensions = document_loaders.keys()  # 支持的文件后缀集合

    # 第一层循环：遍历子目录（如 ai_data、java_data）
    for subject_dir in os.listdir(directory_path):
        subject_path = os.path.join(directory_path, subject_dir)

        # 只处理目录，跳过散落文件
        if not os.path.isdir(subject_path):
            continue

        # 从目录名提取学科来源（如 ai_data → ai，java_data → java）
        source_subject = subject_dir.replace("_data", "")
        logger.info(f"开始处理学科: {source_subject}")

        # os.walk 递归遍历该学科目录下所有层级
        # 每轮产出 (当前目录路径 dirpath, 子目录名列表 dirnames, 文件名列表 filenames)
        for dirpath, dirnames, filenames in os.walk(subject_path):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                # 取小写后缀（.PDF 和 .pdf 都能识别）
                file_ext = os.path.splitext(file_path)[1].lower()

                # 文件类型在支持列表里 → 执行加载
                if file_ext in supported_extensions:
                    try:
                        loader_cls = document_loaders[file_ext]  # 取对应加载器类

                        # TXT 需要指定 UTF-8 编码；其它加载器默认参数即可
                        if file_ext == ".txt":
                            loader = loader_cls(file_path, encoding="utf-8")
                        else:
                            loader = loader_cls(file_path)

                        # 执行加载（返回 Document 列表，通常一份文件一个 Document）
                        loaded_docs = loader.load()

                        # 为每个文档统一添加元数据（用于后续检索过滤与溯源）
                        for doc in loaded_docs:
                            doc.metadata["source"] = source_subject            # 学科来源（过滤用）
                            doc.metadata["file_path"] = file_path              # 文件路径（溯源用）
                            doc.metadata["timestamp"] = datetime.now().isoformat()  # 入库时间

                        documents.extend(loaded_docs)
                        logger.info(f"成功加载文件: {file_path}")
                    except Exception as e:
                        # 单个文件失败不影响其它文件（比如某 PDF 损坏）
                        logger.error(f"加载文件失败: {file_path} | 错误信息: {str(e)}")
                else:
                    # 不支持的类型记一条 warning 跳过
                    logger.warning(f"跳过不支持的文件类型: {file_path}")

    logger.info(f"目录加载完成，总计加载文档片段: {len(documents)}")
    return documents


def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    """
    文档分层分割核心函数（父子块结构）。
    流程：加载文档 → 父块分割 → 子块分割 → 元数据绑定。

    :param directory_path:   文档目录
    :param parent_chunk_size: 父块大小（默认 1200，config.ini 可配）
    :param child_chunk_size:  子块大小（默认 300）
    :param chunk_overlap:     切块重叠长度（默认 50）
    :return: 带完整元数据的"子块"列表（用于向量入库）
    """
    # 1. 加载目录下所有文档（上函数）
    documents = load_documents_from_directory(directory_path)

    # 2. 初始化分割器
    # 普通文本分割器（中文分层分割）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # Markdown 专用分割器（按标题层级分割，结构保留更好）
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    child_chunks = []  # 存储最终分割完成的子块

    # 3. 遍历所有文档，执行分层分割
    for i, doc in enumerate(documents):
        file_path = doc.metadata.get("file_path", "")
        file_ext = os.path.splitext(file_path)[1].lower()

        # 根据文件类型选择分割器：.md 用 Markdown 分割器，其它用中文递归分割器
        is_markdown = (file_ext == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter

        logger.info(f"处理文档: {file_path}, 分割器类型: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 4. 分割父块（大块）
        parent_docs = parent_splitter_to_use.split_documents([doc])

        # 5. 遍历父块，分割为子块并绑定元数据
        for j, parent_doc in enumerate(parent_docs):
            # 生成父块唯一 ID（doc_3_parent_7 这种格式，可读性好且全局唯一）
            parent_id = f"doc_{i}_parent_{j}"

            # 6. 父块分割为子块（小块）
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            # 7. 给每个子块注入元数据：
            #    - parent_id      ：归属哪个父块（去重、回溯用）
            #    - parent_content ：父块完整原文（命中后直接取它喂给大模型！）
            #    - id             ：子块自己的编号
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)

    logger.info(f"分层分割完成，最终子块总数: {len(child_chunks)}")
    return child_chunks


# ====================== 测试入口 ======================
if __name__ == '__main__':
    # 执行文档处理（加载+分割），然后打印切块数量
    chunks = process_documents(r'E:\py_demo\code06\integrated_qa_system\rag_qa\data')
    print("最终生成文档切块数量：", len(chunks))
