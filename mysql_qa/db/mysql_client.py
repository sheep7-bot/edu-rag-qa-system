# -*- coding: utf-8 -*-
"""
mysql_qa/db/mysql_client.py —— MySQL 数据库操作客户端
================================================================

【这个文件负责什么？】
   项目里所有"要落库/查库"的事情都从这里走，共两大类：
     1. FAQ 题库查询：从 jpkb 表拿"所有问题"（给 BM25 建索引）、按问题拿"答案"；
     2. 会话记忆存取：把每轮问答存进 conversations 表，实现多轮对话历史。

【技术 1：PyMySQL】
   Python 操作 MySQL 最常用的库（纯 Python 实现，pip install pymysql）。
   核心三步走（记住这个模板，到哪都能用）：
     1) pymysql.connect(...)     建立连接
     2) connection.cursor()      创建游标（"执行 SQL 的手"）
     3) cursor.execute(sql, 参数) 执行 SQL
        cursor.fetchall()/fetchone() 取结果
   改数据（INSERT/DELETE/UPDATE）后要 connection.commit() 提交，否则不生效！

【技术 2：参数化查询防 SQL 注入（重要安全点）】
   错误写法：cursor.execute(f"SELECT * FROM t WHERE q='{q}'")
             → 用户输入 ' OR '1'='1 就能把表拖走（SQL 注入攻击）
   正确写法（本项目）：cursor.execute("SELECT * FROM t WHERE q=%s", (q,))
             → 用 %s 占位，驱动自动转义，安全。

【怎么打开 MySQL 并使用？】
   1. 安装 MySQL 8.x（或 MariaDB），记住 root 密码（本项目 config.ini 里是 123456）
   2. 启动服务（Windows 服务里 MySQL80 启动即可）
   3. 导入本项目的建库脚本（只做一次）：
        在 cmd 里执行：
          mysql -u root -p < E:\\py_demo\\code06\\integrated_qa_system\\subjects_kg.sql
        或打开 Navicat / DataGrip / MySQL Workbench，
        新建数据库选"运行 SQL 文件"，选择 subjects_kg.sql 执行。
   4. 验证数据：SHOW DATABASES; USE subjects_kg; SHOW TABLES;
        SELECT COUNT(*) FROM jpkb;             -- 应能看到大量 FAQ 问题
        SELECT COUNT(*) FROM conversations;    -- 会话历史表
"""
import pymysql  # Python 的 MySQL 客户端库
import pandas as pd  # 数据分析库（本文件暂未使用，保留导入）
import os
import sys

# 把项目根目录加入 sys.path，保证以"脚本方式"直接运行时也能 import base.*
# （Python 找包是按 sys.path 里的目录找的；__file__ 是当前文件路径，往上三级是项目根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base.config import config  # 数据库连接配置
from base.logger import logger  # 统一日志


class MySQLClient:
    """MySQL 客户端：一个实例持有一个长连接（简化实现为单连接）。"""

    def __init__(self):
        """初始化数据库连接。"""
        try:
            # 创建 MySQL 连接
            # charset='utf8mb4'：支持 emoji 和全部中文（utf8 只有 3 字节，不够）
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                charset='utf8mb4'
            )
            # 创建游标对象，用于执行 SQL 语句
            self.cursor = self.connection.cursor()
            logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            logger.error(f"MySQL 连接失败: {e}")
            # 连接失败主动抛出异常：数据库都没有，程序继续跑没有意义
            raise

    # ==================== 一、FAQ 题库查询 ====================

    def fetch_questions(self):
        """
        查询并返回数据库中所有问题列表（BM25 的语料库）。

        返回：一维问题字符串列表，如 ["2026AI智能应用课程大纲", "学费多少", ...]
        """
        try:
            # 查询题库表所有 question 字段
            self.cursor.execute("SELECT question FROM jpkb")
            # fetchall() 返回格式：[(问题1,), (问题2,), ...] 嵌套元组
            res = self.cursor.fetchall()
            logger.info(f"获取到 {len(res)} 条问题")

            # 遍历结果元组，提取第 0 位的 question 文本，转为一维字符串列表
            questions = [row[0] for row in res]
            return questions
        except pymysql.MySQLError as e:
            logger.error(f"问题查询失败: {e}")
            # 异常返回空列表，防止上层 for 遍历直接崩溃
            return []

    def fetch_answer(self, question):
        """
        根据问题精确匹配，查询对应答案。

        参数：question 问题文本（必须与库里存的完全一致才能命中，如 "学费多少"）
        返回：答案字符串；无数据返回 None。
        """
        try:
            # 使用 %s 占位符，防止 SQL 注入（参数用元组 (question,) 传入！）
            self.cursor.execute("SELECT answer FROM jpkb WHERE question = %s", (question,))
            # fetchone() 只获取第一条匹配记录
            res = self.cursor.fetchone()
            # 三元表达式：查到数据取 answer，没有则返回 None
            return res[0] if res else None
        except pymysql.MySQLError as e:
            logger.error(f"答案查询失败: {e}")
            return None

    # =======================================================
    # 二、会话记忆相关（多轮对话的"记忆"存在 MySQL 里）
    # =======================================================

    def fetch_recent_history(self, session_id: str) -> list:
        """
        获取某个会话最近 5 轮对话历史。

        参数：session_id 会话编号（前端生成，同一个人连续提问用同一个 id）
        返回：按时间正序的 [{"question": ..., "answer": ...}, ...]
        """
        try:
            # 先 commit：把当前未提交的事务提交，保证后续查询看到最新数据
            self.connection.commit()
            # SQL 逻辑：按时间倒序取（ORDER BY timestamp DESC），只要最近 5 条（LIMIT 5）
            self.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, 5))
            history = [{"question": row[0], "answer": row[1]} for row in self.cursor.fetchall()]
            return history[::-1]  # 切片反转：倒序取出来的，反转回"按时间正序"
        except pymysql.MySQLError as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """
        新增一轮对话，并保留最近 5 轮（超出删除）。

        参数：session_id 会话编号；question 本轮问题；answer 本轮答案
        返回：更新后的最近 5 轮历史。
        """
        try:
            # 1. 插入新对话
            self.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))

            # 2. 删除超出 5 轮的旧记录（子查询过滤）
            #    逻辑：挑出该会话最新的 5 条 id，其余 DELETE。
            #    MySQL 不允许在 DELETE 的子查询里直接查同一张表，
            #    所以再包一层 SELECT ... AS sub（这是 MySQL 的经典绕过技巧）。
            self.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, 5))

            self.connection.commit()  # 提交事务（INSERT+DELETE 生效）
            return self.fetch_recent_history(session_id)
        except pymysql.MySQLError as e:
            logger.error(f"更新会话历史失败: {e}")
            self.connection.rollback()  # 出错回滚，保证不留下半截数据
            raise

    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话的所有记录。成功 True / 失败 False。"""
        try:
            self.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            self.connection.commit()
            logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            logger.error(f"清除会话历史失败: {e}")
            self.connection.rollback()
            return False

    def clear_all_sessions_history(self) -> bool:
        """清除所有会话的所有记录（前端"清空历史"按钮的后端逻辑）。"""
        try:
            self.cursor.execute("""
                DELETE FROM conversations
            """)
            self.connection.commit()
            logger.info("所有会话历史已清除")
            return True
        except pymysql.MySQLError as e:
            logger.error(f"清除所有会话历史失败: {e}")
            self.connection.rollback()
            return False

    def fetch_all_sessions(self) -> list:
        """
        获取所有会话列表（按最新时间排序）。

        返回：[{"session_id": ..., "count": 轮数, "last_time": "2026-.. .."}, ...]
        前端左侧"历史会话列表"就是渲染这个数据。
        """
        try:
            # GROUP BY session_id：每个会话聚合成一行，
            # COUNT(*) 统计轮数，MAX(timestamp) 取最后一次对话时间
            self.cursor.execute("""
                SELECT session_id, COUNT(*) as count, MAX(timestamp) as last_time
                FROM conversations
                GROUP BY session_id
                ORDER BY last_time DESC
            """)
            sessions = [{
                "session_id": row[0],
                "count": row[1],
                # datetime 对象转成固定格式字符串，方便前端显示；
                # 三元表达式处理 last_time 为 NULL 的极端情况
                "last_time": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else ""
            } for row in self.cursor.fetchall()]
            return sessions
        except pymysql.MySQLError as e:
            logger.error(f"获取会话列表失败: {e}")
            return []

    def get_session_count(self, session_id: str) -> int:
        """获取指定会话的对话总轮数（cookie 中读取）。"""
        try:
            self.cursor.execute("""
                SELECT COUNT(*) 
                FROM conversations 
                WHERE session_id = %s
            """, (session_id,))
            total_count = self.cursor.fetchone()[0]  # fetchone 拿一行元组，[0] 取数字
            return total_count
        except pymysql.MySQLError as e:
            logger.error(f"获取会话总数失败: {e}")
            return 0

    def close(self):
        """安全关闭游标和数据库连接，释放服务器资源。"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            logger.error(f"连接关闭失败: {e}")


# ============================================================
# 【单独调试】python -m mysql_qa.db.mysql_client
# 会遍历题库的每个问题并打印答案，验证数据库连通性。
# ============================================================
if __name__ == '__main__':
    # 实例化数据库客户端
    client = MySQLClient()

    # 循环读取每条问题，查询并打印对应答案
    for single_question in client.fetch_questions():
        print(f"问题：{single_question}")
        print(f"答案：{client.fetch_answer(single_question)}")
        print("-------")

    # 程序结束，释放数据库连接
    client.close()
