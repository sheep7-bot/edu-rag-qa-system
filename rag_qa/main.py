# -*- coding: utf-8 -*-
"""
rag_qa/main.py —— RAG 问答命令行入口
================================================================

【这个文件干什么的？】
   一个最简命令行界面：调用 IntegratedQASystem（FAQ+RAG 双层链路），
   流式打印答案。适合不开浏览器时快速测试整套后端。

【和 mysql_qa/main.py 的区别】
   - mysql_qa/main.py：只测 FAQ 快通道；
   - 本文件：FAQ 未命中会自动转 RAG 大模型，是完整链路。
   Web 版入口是项目根目录的 app.py（FastAPI + WebSocket + 网页）。

【跑法】
   1. 确认 MySQL、Redis、Milvus 已启动；向量库已建好（跑过 vector_store.py）；
   2. 在项目根目录执行：python -m rag_qa.main
   3. 按提示输入问题、学科过滤；输入 exit 退出。
"""
from rag_qa.core.faq_milvus import IntegratedQASystem


if __name__ == "__main__":
    print("===== 问答系统启动 | exit退出 =====")
    qa = IntegratedQASystem()
    # 输入会话 ID（直接回车用默认 "123"）：同一 ID 的对话会带历史记忆
    session_id = input("会话ID(回车默认123)：").strip() or "123"
    while True:
        q = input("请输入问题：").strip()
        if q.lower() == "exit":
            break
        # 学科过滤：如输入 ai 就只在 AI 知识库里检索；回车=不限
        subj = input("学科名称(回车不限定)：").strip() or None

        print("✅ 答案：", end="", flush=True)
        # qa.query 是生成器：逐片段打印，实现流式打字机效果
        for chunk, finished in qa.query(q, subj, session_id):
            print(chunk, end="", flush=True)
            if finished:  # 收到结束信号
                break
        print("\n")
