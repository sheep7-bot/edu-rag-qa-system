# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/edu_pdfloader.py —— PDF 文档加载器（文字 + 图片OCR）
================================================================

【这个文件干什么的？】
   把一份 PDF 变成 LangChain Document 对象：正文文字直接抽取；
   页面里的大图片（如扫描页、架构图）额外做 OCR 提取文字。

【技术 1：PyMuPDF（导入名 fitz）】
   高性能 PDF 处理库（pip install PyMuPDF）。
     - fitz.open(path)              打开 PDF
     - page.get_text("text")        抽取页面文字
     - page.get_image_info(xrefs=True) 拿页面上所有图片的元信息
   ⚠️ 注意：安装包名是 PyMuPDF，但 import 名字是 fitz；
      不要去 pip install fitz（那是另一个同名老包，会冲突）。

【技术 2：父类 BaseLoader / lazy_load】
   LangChain 自定义加载器的标准写法：
     - 继承 langchain_core.document_loaders.BaseLoader；
     - 实现 lazy_load() 生成器，yield 一个个 Document；
     - 父类会自动帮你实现 load()（把所有 Document 收集成列表）。
   所以外部调用 loader.load() 或 loader.lazy_load() 都可以。

【技术 3：OCR 阈值过滤（性能优化点）】
   一页 PDF 里可能有 logo、图标等小图片，全部 OCR 非常慢且没意义。
   下面只对"尺寸超过页面 60% 的大图"做 OCR（PDF_OCR_THRESHOLD），
   小图标直接跳过，这是非扫描版 PDF 提速的关键技巧。

【跑法】
   直接运行本文件即可测试（需改 txt/PDF 路径）：
        python -m rag_qa.edu_document_loaders.edu_pdfloader
   真正批量入库的入口是 rag_qa/core/document_processor.py。
"""
import cv2            # OpenCV：图像旋转等处理
import fitz           # PyMuPDF：解析 PDF（见上方说明）
import numpy as np    # 数组运算
from PIL import Image  # 图像处理
from tqdm import tqdm  # 进度条
from typing import Iterator
from rag_qa.edu_document_loaders.edu_ocr import get_ocr  # OCR 工厂函数（见 edu_ocr.py）
from langchain_core.documents import Document           # LangChain 文档对象
from langchain_core.document_loaders import BaseLoader  # 自定义加载器的基类
from langchain_text_splitters import CharacterTextSplitter  # 字符分割器（本文件测试用）
# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)


class OCRPDFLoader(BaseLoader):
    """PDF 加载器：解析文字 + 对大图做 OCR。"""

    def __init__(self, file_path: str) -> None:
        """初始化：记住要加载的文件路径。"""
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        惰性加载：yield 一个个 Document（本实现整份 PDF 合成一个 Document）。
        父类 load() 内部会调用本方法并把结果收集成列表。
        """
        # 调用核心解析方法，拿到全文文本
        line = self.pdf2text()
        # 包成 LangChain 标准 Document：page_content 存正文，metadata 存来源（溯源用）
        yield Document(page_content=line, metadata={"source": self.file_path})

    def pdf2text(self):
        """核心方法：遍历每一页，抽文字 + 对大图 OCR。"""
        ocr = get_ocr()          # 获取 OCR 识别器
        # 打开 PDF 文件
        doc = fitz.open(self.file_path)
        ## 获取页数：len(doc) 就是总页数
        resp = ""                # 累积全文文本
        # tqdm：进度条，total=总页数，desc=进度条左侧说明文字
        b_unit = tqdm(total=doc.page_count, desc="OCRPDFLoader context page index: 0")

        # enumerate(doc)：逐页遍历，i 是页码（从 0 开始），page 是页面对象
        for i, page in enumerate(doc):
            b_unit.set_description("OCRPDFLoader context page index: {}".format(i))  # 更新说明
            b_unit.refresh()  # 刷新进度条显示

            # 提取文本：默认使用 "text" 模式提取文本（PDF 里"真实存在"的文字层）
            text = page.get_text("text")
            resp += text + "\n"  # 拼进全文

            # 获取图片：获得所有显示的图像的元信息列表（适用于所有文档类型，不仅限 PDF）
            img_list = page.get_image_info(xrefs=True)

            for img in img_list:
                # xref 是一种编号，指向该图像对象在 PDF 文件中的位置，
                # 程序可以通过这个编号快速定位和提取图像数据。
                # 海象运算符 := ：赋值并同时做条件判断（xref 为空则跳过）
                if xref := img.get("xref"):
                    # bbox 是图像在页面上的位置和尺寸：(x0, y0, x1, y1)
                    bbox = img["bbox"]

                    # 检查图片尺寸是否超过设定的阈值：
                    # 图片宽/页面宽 或 图片高/页面高 < 0.6 就 continue（跳过不做 OCR）
                    if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                        continue

                    # 用 xref 把图片像素数据提取出来
                    pix = fitz.Pixmap(doc, xref)

                    if int(page.rotation) != 0:
                        # 如果页面有旋转角度，则先把页面像素旋转回正向，再 OCR（否则识别率差）
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        tmp_img = Image.fromarray(img_array)
                        ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)  # RGB→BGR（OpenCV 习惯）
                        rot_img = self.rotate_img(img=ori_img, angle=360 - page.rotation)  # 反向旋转校正
                        img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                    else:
                        # 无旋转：直接把 Pixmap 的像素数据转成 numpy 数组
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                    # 调用 OCR：
                    #   result：图像中检测到的所有文本框信息 [框坐标, 文字, 置信度]
                    #   _     ：包含时间数据的列表（耗时），用不上
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]  # 每行取第 1 位=识别文字
                        resp += "\n".join(ocr_result)              # 拼进全文

            # 更新进度条（完成一页）
            b_unit.update(1)
        return resp  # 返回整份 PDF 的文本

    def rotate_img(self, img, angle):
        '''
        图像旋转工具函数。
        img   —— 待旋转图像（numpy 数组）
        angle —— 旋转角度（正=逆时针，负=顺时针）
        return —— 旋转后的图像
        '''

        h, w = img.shape[:2]              # 原图高、宽
        rotate_center = (w / 2, h / 2)    # 旋转中心=图像中心
        # 获取旋转矩阵
        # 参数1为旋转中心点；
        # 参数2为旋转角度，正值-逆时针旋转；负值-顺时针旋转
        # 参数3为各向同性的比例因子，1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算旋转后的新边界（旋转后画布要变大，否则四角会被裁掉）
        new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
        new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
        # 调整旋转矩阵以考虑平移（把旋转后的图像挪回画布中央）
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # warpAffine：按仿射矩阵 M 对图像做变换，输出尺寸 (new_w, new_h)
        rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
        return rotated_img


# ============================================================
# 【单独调试】python -m rag_qa.edu_document_loaders.edu_pdfloader
# 会解析 rag_qa/data/ai_data 里的 LLM基础知识.pdf 并打印 Document。
# ============================================================
if __name__ == '__main__':
    from base.config import config
    pdf_loader = OCRPDFLoader(file_path=f"{config.DATA_DIR}/ai_data/LLM基础知识.pdf")
    doc = pdf_loader.load()  # 父类方法：把 lazy_load 的生成器结果收集成列表

    print(type(doc))
    print(doc)
    # 下面几行是原作者的切块试验（已注释）：
    # text_spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    # result = text_spliter.split_documents(doc)
    # print(len(result))
    # print(result[0])
