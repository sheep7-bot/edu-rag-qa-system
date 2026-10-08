# -*- coding: utf-8 -*-
"""
app.py —— Web 后端服务（FastAPI + WebSocket 流式问答）
================================================================

【这个文件是"网页版"的服务器入口】
   浏览器(static/index.html)
        │  HTTP 请求：创建会话、拉历史、清空历史…
        │  WebSocket 长连接：发问题、收流式答案
        ▼
   本文件（FastAPI 服务，端口 8080）
        │
        ▼
   IntegratedQASystem（FAQ+RAG 双层问答）
        │
        ├── BM25Search（MySQL + Redis）
        └── RAGSystem（BERT + Milvus + DeepSeek）

【工具 1：FastAPI 是什么？】
   目前最流行的 Python Web 框架之一，特点：
     - 用"装饰器"定义接口，代码少、可读性好；
     - 自带交互式接口文档（启动后访问 http://localhost:8080/docs）；
     - 原生支持异步（async def），适合大模型这种 I/O 密集场景。
   本项目用它提供 REST 接口 + WebSocket + 静态网页。

【工具 2：WebSocket 是什么？为什么不用普通 HTTP？】
   普通 HTTP 是"一问一答"：请求一次、响应一次，连接就结束。
   WebSocket 建立连接后可以"一直聊"：服务器能主动把消息推给浏览器。
   大模型回答是一个字一个字生成的，用 WebSocket 才能做到"边生成边推送"，
   用户看到打字机效果，而不是干等 10 秒后一次性出现全部答案。

【怎么启动并使用？】
   1. 确认 MySQL、Redis、Milvus 均已启动，且知识库已建好；
   2. 在项目根目录执行：python app.py
      （等价于 uvicorn app:app --host 0.0.0.0 --port 8080）
   3. 浏览器打开：http://localhost:8080  →  出现聊天页面
   4. 接口文档（调试用）：http://localhost:8080/docs
   5. 健康检查：http://localhost:8080/health
"""
# ==================== Web 框架核心 ====================
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends, Request, Response
from fastapi.responses import StreamingResponse, FileResponse  # 文件响应（返回网页）
from fastapi.middleware.cors import CORSMiddleware  # 跨域中间件
from fastapi.staticfiles import StaticFiles         # 静态文件托管
from starlette.websockets import WebSocketDisconnect  # WebSocket 断开异常

# ==================== 系统与工具 ====================
import os
import asyncio  # 异步：把同步生成器挪到线程里跑（见下方注释）
import json
import uuid   # 生成唯一会话 ID
import time   # 统计处理耗时
import re     # 正则匹配问候语

from typing import Optional, List, Dict, Any

# ==================== 业务层依赖 ====================
from pydantic import BaseModel  # 数据校验（本文件目前未真正使用，保留导入）
from rag_qa.core.faq_milvus import IntegratedQASystem  # 双层问答系统（总入口）
from base.config import config                          # 全局配置
from mysql_qa.db.mysql_client import MySQLClient        # 会话历史

# ==================== 应用实例化 ====================
# FastAPI(title, description)：创建应用对象；title/description 会显示在 /docs 文档页
app = FastAPI(title="问答系统API", description="集成FAQ和RAG的智能问答系统")

# 跨域配置（CORS）：
#   浏览器安全策略默认禁止"不同源"的页面调本接口。
#   开发阶段允许所有来源（*），方便前端联调；生产环境应收紧为具体域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 实例初始化 ====================
# 注意：这两个对象在"模块导入时"创建，启动服务就完成初始化：
#   qa_system 会加载 BERT、BGE-M3、重排序模型，连接 Milvus/MySQL/Redis（耗时较长，属正常）
qa_system = IntegratedQASystem()
mysql_client = MySQLClient()

# ============================================================
# 问候语快捷匹配表：不经过 BM25/RAG，正则命中直接返回固定话术。
# 正则 ^ 表示"以...开头"，| 表示"或"，(?:...) 是分组不捕获写法。
# ============================================================
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是智能学习助手，专注于为学生答疑解惑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是智能学习助手，致力于提供 IT 教育相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "我在！我是智能学习助手，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"
    }
]

# 检查日常问候：命中返回固定话术，没命中返回 None
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        # re.match 从字符串开头匹配；IGNORECASE 忽略大小写（Hi/HI 都能命中）
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


# ============================================================
# REST 接口 1：创建新会话
# POST 请求 http://localhost:8080/api/create_session
# 前端点"新对话"时调用，拿到一个 session_id 用于后续所有请求。
# ============================================================
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成 UUID4 唯一会话 ID（如 3f2a...）
    return {"session_id": session_id}  # 返回给前端

# ============================================================
# REST 接口 2：查询历史消息
# GET http://localhost:8080/api/history/{session_id}
# 前端加载某个会话时调用，返回：最近5轮 history + 总轮数
# ============================================================
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录（最近 5 轮）
        history = mysql_client.fetch_recent_history(session_id)
        # 获取该会话的总对话轮数
        total_count = mysql_client.get_session_count(session_id)
        # 返回会话 ID、历史记录和总数
        return {
            "session_id": session_id,
            "history": history,
            "total_count": total_count
        }
    except Exception as e:
        # HTTPException：返回标准 HTTP 错误（状态码 500 + 详情），前端好处理
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

# ============================================================
# REST 接口 3：查询单个会话的对话总数
# ============================================================
@app.get("/api/session_count/{session_id}")
async def get_session_count(session_id: str):
    try:
        total_count = mysql_client.get_session_count(session_id)
        return {"session_id": session_id, "total_count": total_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话数失败: {str(e)}")

# ============================================================
# REST 接口 4：查询所有会话列表（前端左侧"历史会话"）
# ============================================================
@app.get("/api/sessions")
async def get_all_sessions(response: Response):
    try:
        sessions = mysql_client.fetch_all_sessions()
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

# ============================================================
# REST 接口 5：清除所有会话历史（DELETE 方法）
# ============================================================
@app.delete("/api/history")
async def clear_all_history():
    try:
        success = mysql_client.clear_all_sessions_history()
        if success:
            return {"status": "success", "message": "所有历史记录已清除"}
        else:
            raise HTTPException(status_code=500, detail="清除历史记录失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除历史记录失败: {str(e)}")


# ============================================================
# WebSocket 接口：/api/stream —— 流式问答的"主战场"
#
# 消息协议（前后端约定）：
#   浏览器 → 服务器：{"query": "问题", "source_filter": "ai", "session_id": "..."}
#   服务器 → 浏览器：
#     {"type": "start"}                     开始回答
#     {"type": "token", "token": "字"}      回答片段（多次）
#     {"type": "end", "processing_time": 1.2}  回答结束
#     {"type": "error", "error": "..."}     出错
# ============================================================
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    # 接受客户端建立 WebSocket 长连接（握手）
    await websocket.accept()
    try:
        # 循环监听客户端消息，理论上支持同一连接多次发送提问
        while True:
            # 接收前端发来的 JSON 字符串消息
            data = await websocket.receive_text()
            # 字符串转为字典，提取请求参数
            request_data = json.loads(data)
            # 用户提问内容
            query = request_data.get("query")
            # 知识库数据源过滤条件，用于限定检索范围（如 "ai"）
            source_filter = request_data.get("source_filter")
            # 会话ID：前端传则复用，不传自动生成 UUID 作为会话唯一标识
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            # 记录本轮问答起始时间，用于计算整体耗时
            start_time = time.time()

            # 参数校验：提问为空/全空格，返回错误并继续等下一句（continue 不关连接）
            if not query or not query.strip():
                await websocket.send_json({"type": "error", "error": "查询内容不能为空"})
                continue

            # 向前端发送开始标记，告知前端准备接收流式回答
            await websocket.send_json({"type": "start", "session_id": session_id})

            # ========== 分支1：问候语快捷匹配逻辑 ==========
            # 校验用户输入是否为你好、在吗、你是谁等固定问候语
            greeting_response = check_greeting(query)
            if greeting_response:
                # 将用户问题+AI 固定问候回答存入 MySQL 会话历史表
                mysql_client.update_session_history(session_id, query, greeting_response)
                # 逐字符推送问候文本，模拟打字机输出效果
                for char in greeting_response:
                    await websocket.send_json({
                        "type": "token",
                        "token": char,
                        "session_id": session_id
                    })
                    # 让出事件循环，保证每个字符立即刷到浏览器（否则也会攒批）
                    await asyncio.sleep(0.01)
                # 全部字符推送完成，发送结束标识，附带总耗时
                await websocket.send_json({
                    "type": "end",
                    "session_id": session_id,
                    "is_complete": True,
                    "processing_time": time.time() - start_time
                })
                # 本轮问答结束，跳出 while 循环；当前代码逻辑会进入 finally 关闭连接
                break

            # ========== 分支2：正式 RAG 知识库问答业务逻辑 ==========
            try:
                # qa_system.query 是同步生成器（内部有阻塞的检索/LLM 调用），
                # 必须放到线程里逐个取 token，否则会卡死事件循环，流式被攒成一次性输出。
                # asyncio.to_thread 会把"阻塞调用"丢到线程池执行，不堵住异步主循环。
                gen = qa_system.query(query, source_filter=source_filter, session_id=session_id)

                # 小工具：帮我们安全地"取生成器的下一项"，
                # StopIteration（生成器取完）时返回 None 而不是抛异常。
                def _next_item(g):
                    try:
                        return next(g)
                    except StopIteration:
                        return None

                while True:
                    # 在线程里执行 next(gen)（同步阻塞），asyncio 主循环不受影响
                    item = await asyncio.to_thread(_next_item, gen)
                    if item is None:
                        break  # 生成器结束
                    token, is_complete = item  # 解包：(文本片段, 是否结束)
                    # 推送单段回答分片给前端
                    if token:
                        await websocket.send_json({
                            "type": "token",
                            "token": token,
                            "session_id": session_id
                        })
                    # 生成器标记回答全部输出完毕，发送结束包并终止循环
                    if is_complete:
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                        break
            # RAG 检索/大模型生成过程中出现异常，向前端返回错误信息
            except Exception as query_error:
                await websocket.send_json({"type": "error", "error": f"查询处理失败: {str(query_error)}"})
                # 异常后终止本轮问答，跳出循环关闭连接
                break
    # 捕获客户端主动断开连接事件（关闭页面、刷新、手动关闭 ws）
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    # 捕获其余所有未知异常
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        # 尝试向前端推送错误提示；连接已断开时发送会报错，用 try 忽略
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    # 无论正常结束、异常、客户端断开，最终都会执行，关闭 ws 连接释放资源
    finally:
        try:
            await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {str(e)}")


# ============================================================
# 健康检查接口：运维/负载均衡用来确认"服务还活着"
# GET http://localhost:8080/health  →  {"status": "healthy"}
# ============================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ============================================================
# 获取学科类别：前端"学科过滤"下拉框的数据来源
# GET http://localhost:8080/api/sources → {"sources": ["ai","java",...]}
# ============================================================
@app.get("/api/sources")
async def get_sources():
    return {"sources": config.VALID_SOURCES}


# ============================================================
# 挂载静态文件服务：把 static/ 目录映射到 /static 路径
# 例如文件 static/index.html 可通过 http://localhost:8080/static/index.html 访问
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")

# 首页：直接返回聊天页面 static/index.html
@app.get("/")
async def read_root():
    return FileResponse(
        "static/index.html",
        # 加这三个响应头禁用浏览器缓存：改完前端刷新即生效，避免缓存旧页面
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


# ============================================================
# 【启动方式】
#   方式一（本项目）：python app.py
#   方式二（推荐正式环境）：uvicorn app:app --host 0.0.0.0 --port 8080
#   参数说明：
#     host="0.0.0.0" → 允许局域网其它机器访问（只本机访问可写 127.0.0.1）
#     port=8080      → 端口
#     reload=False   → 改代码是否自动重启；学习调试可改 True（但每次重启都会
#                      重新加载 BERT/向量模型，非常慢，建议保持 False）
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
