# -*- coding: utf-8 -*-
"""
mysql_qa/cache/redis_client.py —— Redis 缓存客户端
================================================================

【什么是 Redis？为什么用它？】
   Redis 是一个"内存数据库"——数据放内存里，读写速度是 MySQL 的几十上百倍。
   本项目中它有两大用途：
     1. 缓存题库和答案：BM25 每次启动都要全量加载问题，
        放 Redis 里就不用反复查 MySQL（这叫"缓存/冷热分层"）；
     2. 答案快速命中：一模一样的用户问题直接从缓存拿答案，跳过一切计算。

【技术概念：JSON 序列化】
   Redis 只能存字符串/字节。要存字典、列表这些复杂结构，
   需要先转成 JSON 字符串（json.dumps），取出来再转回 Python 对象（json.loads）。
   ensure_ascii=False 让中文原样存储，方便直接在 Redis 客户端里看懂。

【怎么打开 Redis 并使用？（Windows 学习环境）】
   1. 安装：下载 Redis for Windows（如 tporadowski/redis 发行版）或使用 WSL/Docker；
      Docker 一条命令：docker run -d --name redis -p 6379:6379 redis
   2. 启动：redis-server（Docker 方式则容器启动即服务）
   3. 用命令行客户端连接：redis-cli
   4. 常用命令（学了就能验证本文件存的缓存）：
        KEYS *                 # 看所有键
        GET qa_original_questions   # 看题库缓存
        GET "answer:你好"       # 看某个问题的缓存答案
        FLUSHDB                # 清空当前库（调试时常用）
"""
import redis  # Python 的 Redis 客户端库：pip install redis
import json   # 标准库：JSON 序列化/反序列化工具

from base.config import config  # 读取 Redis 连接配置
from base.logger import logger  # 统一日志


class RedisClient:
    """Redis 客户端封装：连接 + 通用存取 + 问答专用快捷方法。"""

    def __init__(self):
        """
        初始化 Redis 连接。
        从全局配置读取连接参数，自动解码响应为字符串，记录连接状态。
        """
        self.logger = logger
        try:
            # 创建 Redis 连接客户端
            # decode_responses=True：读出来的字节自动解码成 str，
            #   否则 get() 返回 b'...'（bytes），还要手动 .decode()，很麻烦
            self.client = redis.Redis(
                host=config.REDIS_HOST,   # 主机地址（默认 localhost）
                port=config.REDIS_PORT,   # 端口（默认 6379）
                db=config.REDIS_DB,       # 库编号 0~15
                decode_responses=True
            )
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 连接参数错误时（如地址写错）抛异常终止，避免后面用坏连接
            self.logger.error(f"Redis 连接失败: {e}")
            raise

    def set_data(self, key, value):
        """
        通用数据存储：将任意可 JSON 序列化的数据存入 Redis。

        参数：
            key:   缓存键名（如 "qa_original_questions"）
            value: 待存储数据（列表/字典/字符串都行）
        """
        try:
            # json.dumps：Python 对象 → JSON 字符串
            # ensure_ascii=False：中文不转义（否则"你好"会变成 \u4f60\u597d）
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            self.logger.info(f"数据已存储: {key}")
        except redis.RedisError as e:
            # 存缓存失败不算致命错误，记日志继续跑（缓存没了顶多慢一点）
            self.logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        """
        通用数据获取：从 Redis 读取并反序列化 JSON 数据。

        返回：解析后的 Python 对象；键不存在时返回 None。
        """
        try:
            data = self.client.get(key)
            # 三元表达式：有数据就 json.loads 还原，没有就 None
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f"Redis 获取数据失败: {e}")
            return None

    def get_answer(self, query):
        """
        问答专用：根据问题查询缓存答案。

        约定：答案统一用 "answer:问题" 作键名（加前缀方便用 KEYS answer:* 批量查看）。
        返回：缓存答案；未命中返回 None。
        """
        try:
            answer = self.client.get(f"answer:{query}")
            return json.loads(answer) if answer else None
        except redis.RedisError as e:
            self.logger.error(f"Redis 查询失败: {e}")
            return None


# ============================================================
# 【单独调试】python -m mysql_qa.cache.redis_client
# 会往 Redis 写一条测试数据再读出来。
# 调试成功后可用 redis-cli 执行 GET "answer:测试问题" 亲眼验证。
# ============================================================
if __name__ == '__main__':
    redcli = RedisClient()
    query = "测试问题"
    answer = "测试答案"
    redcli.set_data(f"answer:{query}", answer)
    print(redcli.get_answer("测试问题"))
