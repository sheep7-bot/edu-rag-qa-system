# -*- coding: utf-8 -*-
"""
base/logger.py —— 日志管理模块（全项目统一日志出口）
================================================================

【为什么需要日志，不用 print？】
   print 只能打在控制台，程序一关就没了；日志会：
     1. 同时写文件（logs/app.log）+ 打控制台，出问题可事后翻记录；
     2. 带时间、模块名、级别，方便定位"谁在什么时候报了什么错"；
     3. 可通过级别开关控制输出量（DEBUG 调试细节 / INFO 常规 / ERROR 只报错）。

【logging 四个核心概念（背下来）】
   Logger     —— 日志记录器（logger.info("...") 就是它在干活）
   Handler    —— 处理器：决定日志"输出到哪里"（文件/控制台/网络）
   Formatter  —— 格式器：决定每条日志长什么样
   Level      —— 级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
                 Logger 设置 INFO，则 DEBUG 消息被忽略（不输出）。

【本文件的模式：全局单例】
   模块加载时执行一次 setup_logging() 生成 logger，
   其它文件 `from base.logger import logger` 拿到同一个对象，
   避免"每个文件各建一套日志导致重复打印"。
"""
import logging  # Python 标准库：日志
import os       # 处理路径
from base.config import config  # 读取 config.ini 里的日志路径配置

# 项目根目录（绝对路径，保证日志位置固定，不受"从哪个目录启动"影响）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_file=config.LOG_FILE):
    """
    初始化并返回全局 logger。

    参数 log_file：日志文件相对路径（来自 config.ini 的 [logger] 节）。
    """
    # 1) 拼接日志绝对路径 → 强制写到项目根目录下
    log_path = os.path.join(BASE_DIR, log_file)
    # 2) 自动创建日志文件夹（logs/ 不存在时自动建；exist_ok=True 表示已存在不报错）
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 3) 获取全局日志实例（name="EduRAG"，其它库一般用自己名字）
    logger = logging.getLogger("EduRAG")
    # 4) 设置最低级别：DEBUG 及以上全部处理
    logger.setLevel(logging.DEBUG)

    # 5) 避免重复添加处理器：
    #    logging.getLogger 同名返回的是同一个对象，
    #    如果模块被重复加载（reload），不加这行会导致日志打印两遍、三遍……
    if logger.handlers:
        return logger

    # 6) 定义日志格式：
    #    asctime=时间  name=logger名  levelname=级别  message=日志内容
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 7) 文件处理器：日志写进 logs/app.log（encoding='utf-8' 防止中文乱码）
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)  # 文件只记 INFO 以上（不记 DEBUG 细节）

    # 8) 控制台处理器：日志同时打到终端，跑程序时肉眼可见
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 9) 把两个处理器都挂到 logger 上
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# 全局单例日志对象：其它文件直接 `from base.logger import logger` 使用即可
# 用法示例：
#     logger.info("检索完成，共 3 条")
#     logger.error(f"连接失败: {e}")
# ============================================================
logger = setup_logging()
