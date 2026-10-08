# -*- coding: utf-8 -*-
"""
rag_qa/core/query_classifier.py —— 用户问题分类器（BERT 二分类）
================================================================

【这个文件在问答链路里的位置】
   用户问题进来后先过这里，判断是：
     "通用知识"（你好、闲聊、常识） → 走大模型直接答，不用知识库
     "专业咨询"（课程大纲、学费、学科区别） → 必须检索 Milvus 知识库
   这一步叫"查询路由（Query Routing）"，能省检索开销、减少答非所问。

【技术：BERT + 微调（Fine-tuning）】
   BERT 是 Google 出的预训练语言模型，擅长理解语义。
   原始 BERT 不会分辨"通用/专业"，作者用课程数据做了二分类微调，
   得到一个专用小模型，存放在：
       E:/Develop/embedding/bert_query_classifier（Windows 路径）
   本项目直接加载这个"已经训练好的模型"做推理。

【推理四步走（所有 transformers 分类任务都是这个套路）】
   1. tokenizer(...)  ：把文本转成模型能吃的数字张量（id、attention_mask）；
   2. model(**encodings).logits ：前向计算，得每个类别的原始分数；
   3. torch.argmax(...) ：取分数最大的类别下标；
   4. 映射业务标签      ：0 → 通用知识，1 → 专业咨询。

【参数解释（tokenizer 的三个关键参数）】
   truncation=True ：文本超长时截断
   padding=True    ：文本过短时补 0，对齐到统一长度（批处理必需）
   max_length=128  ：序列最长 128 个 token
   return_tensors="pt" ：返回 PyTorch 张量（pt = PyTorch）

【torch.no_grad() 是什么？】
   告诉 PyTorch"只做推理，不计算梯度"。
   训练时要反传梯度，推理时不需要 → 关掉能省内存、提速。

【怎么单独测试？】
   python -m rag_qa.core.query_classifier
   会打印 4 个测试问题的分类结果。
"""
import torch  # PyTorch：深度学习框架
from transformers import BertTokenizer, BertForSequenceClassification  # HuggingFace 模型库
from base.logger import logger


class QueryClassifier:
    """用户问题分类器：通用知识 vs 专业咨询。"""

    def __init__(self):
        # BERT 模型目录（本地微调好的模型，改动这里可换模型）
        model_dir = r"E:\Develop\embedding\bert_query_classifier"
        # 加载 tokenizer：负责"文本 → 数字张量"。与训练时同一套词表，必须配对使用。
        # 如果本地目录打不开，也可以传 "bert-base-chinese" 会自动联网下载
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        # 加载分类模型并切到"评估模式"：
        #   .eval() 会关闭 Dropout 等训练专用层，保证推理结果稳定。
        self.bert_model = BertForSequenceClassification.from_pretrained(model_dir).eval()
        logger.info("问题分类模型加载成功")

    def predict_category(self, query: str):
        """
        预测问题分类。

        :param query: 用户问题字符串
        :return: "通用知识" / "专业咨询"
        """
        # 空输入/非字符串直接兜底为"通用知识"（最安全路径）
        if not query or not isinstance(query, str):
            return "通用知识"

        try:
            # 1) BERT 分词：原始文本转为模型需要的张量输入
            encodings = self.tokenizer(
                query,
                truncation=True,       # 超长截断
                padding=True,          # 不足补占位符，对齐长度
                max_length=128,        # 序列最大 token 长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
            # 2) 模型推理（关闭梯度计算）
            with torch.no_grad():
                #    bert_model(**encodings).logits → 每个类别的原始分数，形状 [1, 2]
                #    argmax(dim=1) → 取第 1 维（类别维）最大值的下标
                #    .item()       → 张量转普通 Python 整数
                prediction = torch.argmax(self.bert_model(**encodings).logits, dim=1).item()

            # 3) 映射业务标签：1=专业咨询，0=通用知识
            return "专业咨询" if prediction == 1 else "通用知识"

        except Exception as e:
            # 推理异常时兜底为"通用知识"，避免整个问答流程崩掉
            logger.error(f"问题分类预测失败: {e}")
            return "通用知识"


# ============================================================
# 【单独调试】python -m rag_qa.core.query_classifier
# 预期：springboot的原理/AI学科学费/AI课程大纲 → 专业咨询；
#       你好 之类 → 通用知识。
# ============================================================
if __name__ == '__main__':
    classifier = QueryClassifier()
    test_queries = [
        "springboot的原理是什么",
        "AI学科的学费是多少",
        "你好",
        "AI学科的的课程大纲"
    ]

    for q in test_queries:
        cat = classifier.predict_category(q)
        print(f"问题：{q} -> 分类：{cat}")
