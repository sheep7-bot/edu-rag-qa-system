# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/edu_imgloader.py —— 图片（.jpg/.png）加载器
================================================================

【这个文件干什么的？】
   最"薄"的加载器：把一张图片交给 OCR，识别出文字并包成 LangChain Document。
   支持的文件类型在 document_processor.py 的映射表里配置：
       ".jpg": OCRIMGLoader, ".png": OCRIMGLoader

【RapidOCR 的输入很灵活】
   既支持传"图片文件路径"（本文件用法），也支持传 numpy 数组（PDF/Word/PPT 里的用法）。
"""
from typing import Iterator
from rag_qa.edu_document_loaders.edu_ocr import get_ocr  # OCR 工厂
from langchain_core.documents import Document            # LangChain 文档对象
from langchain_core.document_loaders import BaseLoader   # 加载器基类


class OCRIMGLoader(BaseLoader):
    """图片加载器：OCR 提取图片文字。"""

    def __init__(self, img_path: str) -> None:
        """初始化：记住图片路径。"""
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载：识别文字并产出 Document。"""
        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        """核心方法：调 OCR，把结果文字拼成字符串。"""
        resp = ""
        ocr = get_ocr()
        # 直接把图片路径传给 RapidOCR（内部自动读图）
        result, _ = ocr(self.img_path)
        if result:
            # result 每项结构：[框坐标, 文字, 置信度] → 取第 1 位文字
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp


# ============================================================
# 【单独调试】python -m rag_qa.edu_document_loaders.edu_imgloader
# 注意把下面路径改成你本机的真实图片路径。
# ============================================================
if __name__ == '__main__':
    img_loader = OCRIMGLoader(img_path='/0001.项目目录/others/test/人工智能就业课课程大纲.png')
    doc = img_loader.load()
    print(doc)
