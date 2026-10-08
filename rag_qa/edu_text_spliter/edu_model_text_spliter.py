# -*- coding: utf-8 -*-
"""
rag_qa/edu_text_spliter/edu_model_text_spliter.py —— 阿里语义分割器（备选方案）
================================================================

【这个文件和 ChineseRecursiveTextSplitter 的区别？】
   - ChineseRecursiveTextSplitter：按"标点符号"切（规则法），快，无需模型；
   - AliTextSplitter（本文件）：用"语义分割模型"判断句子边界（模型法），
     更懂语义，但要跑模型、需要额外依赖 modelscope[nlp]，速度慢。
   本项目主链路用的是规则法；本文件作为进阶备选保留。

【技术：达摩院文档语义分割模型】
   nlp_bert_document-segmentation_chinese-base（ModelScope 开源）
   论文：https://arxiv.org/abs/2107.09278
   作用：给长文档做"语义分段"，输出按段落切好的文本。

【注意点】
   1. 需安装：pip install "modelscope[nlp]"
   2. 模型文件需提前下载到本地，路径见 config.MODELS_DIR；
   3. 代码里 device="cpu" 是为了低配机器也能跑，有 GPU 可改成你的显卡编号。
"""
from langchain_text_splitters import CharacterTextSplitter  # 父类：字符分割器
import re
from typing import List
from modelscope.pipelines import pipeline  # 魔搭的模型流水线入口


class AliTextSplitter(CharacterTextSplitter):
    """阿里语义分割器：重写 split_text，改用语义模型来分段。"""

    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        # pdf 标志：若输入来自 PDF，先做文本清洗（去多余空白/换行）
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        """用语义分割模型把长文本切成段落列表。"""
        # use_document_segmentation 参数指定是否用语义切分文档，
        # 此处采取的文档语义分割模型为达摩院开源的
        # nlp_bert_document-segmentation_chinese-base
        # 如果使用模型进行文档语义切分，那么需要安装 modelscope[nlp]：
        #   pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置 gpu 不太友好，
        # 因此这里将模型 load 进 cpu 计算，有需要的话可以替换 device 为自己的显卡 id
        if self.pdf:
            # 清洗 PDF 文本：3 个以上换行压成 1 个、所有空白压成单空格、去掉双换行
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r'\s', " ", text)
            text = re.sub("\n\n", "", text)

        # 局部导入 config，避免顶层导入时影响包加载
        from base.config import config
        # 创建语义分割流水线（从本地模型目录加载）
        p = pipeline(
            task="document-segmentation",
            model=f'{config.MODELS_DIR}/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        # 执行分割：result["text"] 是按 "\n\t" 分隔的段落串
        result = p(documents=text)
        # 按 "\n\t" 拆开，过滤空串，得到段落列表
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list


if __name__ == '__main__':
    # 测试：直接跑一段中文，观察语义分割结果
    model_split = AliTextSplitter()
    result = model_split.split_text(text='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')
    print(result)
