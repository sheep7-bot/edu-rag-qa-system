# -*- coding: utf-8 -*-
"""
rag_qa/rag_assesment/ragas_evaluate.py —— RAGAS 自动化评估脚本
================================================================

【为什么 RAG 系统需要"评估"？】
   改了一版切块大小、换了提示词，效果到底变好还是变坏？
   靠人肉试几个问题太主观。RAGAS 用大模型当"考官"，
   对每条问答自动打分，改版前后一对比就知道有没有进步。

【工具：RAGAS】
   RAG 评估专用框架（pip install ragas），四大核心指标：
     faithfulness       忠实度：答案里的每句话是否都有检索内容支撑（查幻觉）
     answer_relevancy   答案相关性：答案和问题是否对题（用向量算相似度）
     context_precision  上下文精度：检索回来的内容里，有用内容占比多高
     context_recall     上下文召回：标准答案需要的信息，检索到了没有
   评估原理：RAGAS 自己调用 LLM 对这些问题逐条打分（所以评估也要花 token）。

【数据格式要求】
   每条样本 4 个字段：
     question     用户问题
     answer       你的系统生成的答案
     contexts     系统检索到的上下文列表（注意是"嵌套列表"：[[段1,段2], [段1], ...]）
     ground_truth 人工标准答案（参考答案）

【怎么跑？】
   1. 先准备评估数据：rag_evaluate_data_small.json（小样本，几分钟跑完）
      完整版 rag_evaluate_data.json（样本多，耗时更久）
   2. 触发评估的时机：本地配置好 API Key 后执行：
          cd 项目根
          python -m rag_qa.rag_assesment.ragas_evaluate
   3. 结果：
      - 控制台打印总体分数
      - 每条样本明细保存到 ragas_evaluation_results.csv（可直接用 Excel 打开）
"""
import json  # 读 JSON 数据

# huggingface datasets：把 Python 字典转成标准的 Dataset 数据结构（RAGAS 要求）
from datasets import Dataset
# LLM 初始化、本地 Embedding 模型
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
# RAGAS 评估入口 + 四大评估指标
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # 忠实度，检测回答幻觉
    answer_relevancy,   # 答案相关性，衡量回答与问题匹配程度
    context_precision,  # 上下文精准度，评估检索质量：检索结果有效占比
    context_recall      # 上下文召回率，评估检索质量：标准答案信息是否被检索获取
)

# 项目配置：读取 LLM 接口地址、密钥、模型名称等配置
from base.config import config


def main():
    # ---------------------- 1. 加载评估数据集 ----------------------
    # json 数据集字段：question、answer、context、ground_truth
    # 注意路径是相对"配置文件"写的；如果你从别的目录跑报错，
    # 可改成绝对路径或 os.path.join(os.path.dirname(__file__), ...)
    with open(r"./rag_evaluate_data_small.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------------- 2. 转换为 RAGAS 要求的数据集格式 ----------------------
    # RAGAS Dataset 固定字段：question / answer / contexts / ground_truth
    eval_data = {
        "question": [item["question"] for item in data],         # 用户问题列表
        "answer": [item["answer"] for item in data],             # RAG 模型生成的答案列表
        "contexts": [item["context"] for item in data],          # 检索得到的上下文列表（嵌套列表！）
        "ground_truth": [item["ground_truth"] for item in data]  # 人工标准答案
    }
    # from_dict：由 Python 字典创建 Dataset
    dataset = Dataset.from_dict(eval_data)

    # ---------------------- 3. 初始化评估依赖：评估 LLM + Embedding 向量模型 ----------------------
    # RAGAS 本身需要调用大模型做指标打分，此处使用 OpenAI 兼容接口（DeepSeek）
    llm = init_chat_model(
        model=config.LLM_MODEL,        # 评估用大模型名称
        model_provider="openai",       # 使用 openai 协议
        api_key=config.LLM_API_KEY,    # 接口密钥
        base_url=config.LLM_BASE_URL   # 接口访问地址
    )

    # 使用本地 BGE-M3 向量模型，用于 answer_relevancy 指标计算
    # （RAGAS 要把"答案"和"问题"都转向量算相似度）
    embeddings = HuggingFaceEmbeddings(
        model_name=r"E:\Develop\embedding\bge-m3"
    )

    # ---------------------- 4. 执行 RAGAS 评估 ----------------------
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,               # RAGAS 打分所使用的大模型
        embeddings=embeddings  # RAGAS 打分所使用的向量模型
    )

    # ---------------------- 5. 打印结果并保存 csv 明细 ----------------------
    print(f"===== RAGAS整体评估结果 =====\n{result}")

    # 转为 pandas 表格：每一行对应一条测试样本的各项分数
    result_df = result.to_pandas()
    # utf-8-sig：带 BOM 的 utf-8，Excel 打开不乱码
    result_df.to_csv("ragas_evaluation_results.csv", index=False, encoding="utf-8-sig")
    print("✅ 每条样本详细评估结果已保存至 ragas_evaluation_results.csv")


if __name__ == "__main__":
    main()
