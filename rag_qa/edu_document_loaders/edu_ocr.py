# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/edu_ocr.py —— OCR 工厂函数（统一获取文字识别器）
================================================================

【什么是 OCR？】
   OCR = Optical Character Recognition，光学字符识别：
   把"图片里的文字"识别成"可编辑的文本"。
   本项目的 PDF 扫描页、Word 截图、PPT 图片、独立图片，
   都靠 OCR 变成文本，才能进入后续的切块和向量化流程。

【工具：RapidOCR】
   本项目用 RapidOCR（基于 PaddleOCR 模型的轻量封装）：
     - rapidocr_paddle        ：GPU/CPU 均可，GPU 下更快（需要装 paddlepaddle）
     - rapidocr_onnxruntime   ：只有 CPU 时的首选，体积小、资源占用低
   下面的 try/except 就是"优先 GPU 版，装不上就退到 CPU 版"的自动降级写法。

【返回值的结构（重要，后面几个 loader 都用到）】
   result 形如：[
       [[文本框左上角坐标...], "识别出的文字", 置信度],
       ...
   ]
   所以后文常见写法 result[i][1] 取文字、result[i][2] 是置信度。
   第二个返回值 _ 是耗时信息，用不上所以用下划线接住。
"""
from typing import TYPE_CHECKING  # 仅用于类型注解提示，运行时不真正导入

'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。它在 CPU 上进行了优化，资源占用较低.
'''

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    """
    获取 OCR 识别器实例（工厂函数）。

    参数：
        use_cuda: 是否使用 GPU。
                  True  → 三个子模型（检测/方向分类/识别）都尝试用 GPU；
                  False → 纯 CPU 模式。
    返回：
        RapidOCR 实例，调用方式：result, elapse = ocr(图片路径或numpy数组)
    """
    try:
        # 首选：GPU 版（paddle 引擎）
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        # 降级方案：onnxruntime 版（没装 GPU 版也能跑，纯 CPU）
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
