# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/edu_pptloader.py —— PowerPoint（.ppt/.pptx）加载器
================================================================

【这个文件干什么的？】
   把 PPT 每一页的文字框、表格、图片（OCR）全部提取出来，拼成纯文本。

【技术：python-pptx】
   prs = Presentation(path)     打开 PPT
   prs.slides                   幻灯片集合（可遍历）
   slide.shapes                 一页上的所有"形状"（文本框/表格/图片/组合...）
   形状类型 shape.shape_type：
     13 = PICTURE（图片）   → 需要 OCR
     6  = GROUP（组合）     → 里面还嵌套着子形状，要递归
     shape.has_text_frame   → 有文本框就取 shape.text
     shape.has_table        → 有表格就遍历单元格

【排序细节（提升阅读顺序）】
   sorted(slide.shapes, key=lambda x: (x.top, x.left))
   按"上→下、左→右"排序形状，让提取出的文字顺序接近人眼阅读顺序。
   （PPT 里形状的存储顺序和视觉顺序经常不一致）

【递归处理组合形状】
   组合形状（GROUP）里面可能还有文本框、图片、甚至孙级组合，
   所以 extract_text 用递归函数处理每一层。
"""
from typing import Iterator
from rag_qa.edu_document_loaders.edu_ocr import get_ocr  # OCR 工厂
from langchain_core.documents import Document            # LangChain 文档对象
from langchain_core.document_loaders import BaseLoader   # 加载器基类
from pptx import Presentation                            # python-pptx 入口
from PIL import Image                                    # 图片处理
import numpy as np                                       # 数组
from io import BytesIO                                   # 内存字节流
from tqdm import tqdm                                    # 进度条


class OCRPPTLoader(BaseLoader):
    """PPT 加载器：解析文字框 + 表格 + 图片 OCR。"""

    def __init__(self, filepath: str) -> None:
        """初始化：记住要加载的文件路径。"""
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载：整份 PPT 合成一个 Document 输出。"""
        line = self.ppt2text(self.filepath)
        yield Document(page_content=line, metadata={"source": self.filepath})

    def ppt2text(self, filepath):
        """核心方法：逐页遍历，提取所有可见文字。"""
        # 打开指定路径的 PowerPoint 文件
        prs = Presentation(filepath)
        # 获取 OCR 功能的实例
        ocr = get_ocr()
        # 初始化一个空字符串，用于存储提取的文本内容
        resp = ""

        def extract_text(shape):
            """
            递归提取单个形状里的文字。
            参数 shape：一个形状对象（文本框/表格/图片/组合）。
            """
            # nonlocal 声明：表示 resp 是"外层函数的变量"，
            # 允许内层函数直接修改外层的 resp（否则会当成新建局部变量报错）
            nonlocal resp

            # ---- 情况 1：形状是文本框 ----
            if shape.has_text_frame:
                # 把文本框中的文本拼进结果，并去掉前后空白
                resp += shape.text.strip() + "\n"

            # ---- 情况 2：形状是表格 ----
            if shape.has_table:
                # 遍历表格的每一行
                for row in shape.table.rows:
                    # 遍历每一行中的每个单元格
                    for cell in row.cells:
                        # 遍历单元格中的每个段落
                        for paragraph in cell.text_frame.paragraphs:
                            # 单元格文字拼进结果
                            resp += paragraph.text.strip() + "\n"

            # ---- 情况 3：形状是图片（shape_type == 13）----
            if shape.shape_type == 13:  # 13 表示图片
                # 图片的二进制数据 → BytesIO → PIL 图像
                image = Image.open(BytesIO(shape.image.blob))
                # 转 numpy 数组交给 OCR
                result, _ = ocr(np.array(image))
                if result:  # OCR 有结果
                    # 提取每行文字（第 1 位）
                    ocr_result = [line[1] for line in result]
                    # 拼进结果，以换行分隔
                    resp += "\n".join(ocr_result)

            # ---- 情况 4：形状是组合（shape_type == 6）----
            elif shape.shape_type == 6:  # 6 表示组合
                # 组合里还有子形状，递归调用自己逐个处理
                for child_shape in shape.shapes:
                    extract_text(child_shape)

        # 创建进度条：总共 len(prs.slides) 页
        b_unit = tqdm(total=len(prs.slides), desc="OCRPPTLoader slide index: 1")

        # 遍历所有幻灯片（start=1 让编号从 1 开始，符合人的习惯）
        for slide_number, slide in enumerate(prs.slides, start=1):
            b_unit.set_description("OCRPPTLoader slide index: {}".format(slide_number))
            b_unit.refresh()  # 刷新进度条

            # 按"从上到下、从左到右"排序形状（x.top 纵坐标优先，x.left 横坐标其次）
            sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))

            for shape in sorted_shapes:
                extract_text(shape)  # 提取当前形状的文字

            b_unit.update(1)  # 一页处理完，进度 +1

        return resp  # 返回提取到的所有文本


# ============================================================
# 【单独调试】python -m rag_qa.edu_document_loaders.edu_pptloader
# ============================================================
if __name__ == '__main__':
    from base.config import config
    img_loader = OCRPPTLoader(filepath=f'{config.DATA_DIR}/ai_data/01.pptx')
    doc = img_loader.load()
    print(doc)
