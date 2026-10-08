# -*- coding: utf-8 -*-
"""
base/config.py —— 项目配置管理模块（读 config.ini）
================================================================

【这个文件在项目中的位置】
   整个项目的"总控台"：所有模块（MySQL、Redis、Milvus、大模型、检索参数）
   都从这里拿配置。你只需要改根目录的 config.ini，全项目自动生效。

【知识点 1：为什么配置文件不直接写在代码里？】
   用户名/密码/地址这类"环境相关"的信息应该和代码分离：
     - 换台电脑部署，只改 config.ini，不用动一行代码；
     - 密码不进 Git 仓库（生产项目会把 config.ini 加入 .gitignore）。

【知识点 2：configparser 是什么？】
   Python 标准库自带的 INI 文件解析器。
   INI 文件长这样（见项目根目录 config.ini）：
       [mysql]
       host = localhost
       user = root
   中括号叫"节（section）"，host/user 叫"键（key）"。
   config.get('mysql', 'host') 就是取 [mysql] 节下的 host 值。

【知识点 3：为什么到处写 fallback=？】
   fallback 是"读不到配置时的兜底默认值"。
   这样即使 config.ini 缺了某一项，程序也不会崩，而是用默认值继续跑。

【知识点 4：os.getenv(...) 与 os.environ 是什么？】
   环境变量是操作系统级别的"全局小纸条"。
   写法 `os.getenv("X", 默认值)` 表示：优先读环境变量 X，读不到用默认值。
   好处：同一个 config.ini 在服务器上不用写死密码，
   部署时用环境变量覆盖即可（Docker/K8s 部署的通行做法）。
"""
import configparser  # Python 标准库：解析 .ini 配置文件
import os            # 处理路径、读环境变量

# ============================================================
# 项目根目录绝对路径（基于当前文件向上两级目录）
# 当前文件: 项目根/base/config.py
#   os.path.abspath(__file__)      → .../项目根/base/config.py
#   os.path.dirname(第1次)          → .../项目根/base
#   os.path.dirname(第2次)          → .../项目根
# 作用：无论你在哪个目录下执行 python，都能正确找到 config.ini，
# 解决"不同运行环境下相对路径读取失败"的经典问题。
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """配置类：把所有配置项读进来，变成对象的属性（config.MYSQL_HOST 这样用）。"""

    def __init__(self, config_file='config.ini'):
        # 拼接 config.ini 的绝对路径
        self.config_path = os.path.join(BASE_DIR, config_file)

        # 创建配置解析器（可以理解为"读 ini 的机器"）
        self.config = configparser.ConfigParser()
        # 读取配置文件（指定 utf-8，防止中文注释乱码）
        self.config.read(self.config_path, encoding='utf-8')

        # ===================== MySQL 配置 =====================
        # 对应 config.ini 的 [mysql] 节
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')  # 主机地址
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')       # 用户名
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='root')  # 密码
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')  # 数据库名

        # ===================== Redis 配置 =====================
        # getint：把读到的字符串转成整数（port、db 都必须是 int）
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')  # 主机地址
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)      # 端口
        # self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='root')  # 密码（按需开启）
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)             # 数据库编号（0~15）

        # ===================== 日志配置 =====================
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')  # 日志文件路径

        # ===================== Milvus 向量库配置 =====================
        # Milvus：开源的向量数据库，专门存"向量"并做相似度检索（RAG 的检索引擎）
        # 这里也支持用环境变量覆盖，方便服务器部署
        self.MILVUS_URI = os.getenv("MILVUS_PORT", self.config.get("milvus", "uri", fallback="http://localhost:19530"))
        self.MILVUS_DATABASE_NAME = os.getenv("MILVUS_DATABASE_NAME",
                                              self.config.get("milvus", "database_name", fallback="itcast"))
        self.MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME",
                                                self.config.get("milvus", "collection_name", fallback="edurag_0421"))

        # ===================== LLM 大模型配置 =====================
        # 本项目用 DeepSeek 的 OpenAI 兼容接口作为生成模型
        self.LLM_MODEL = self.config.get("llm", "model", fallback="deepseek-v4-flash")
        self.LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", self.config.get("llm", "llm_api_key"))
        self.LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL",
                                      self.config.get("llm", "llm_base_url", fallback="https://api.deepseek.com"))

        # ===================== 检索分块参数 =====================
        # 父子分块：父块大（1200字）给大模型当上下文，子块小（300字）用于精准检索
        self.PARENT_CHUNK_SIZE = self.config.getint("retrieval", "parent_chunk_size", fallback=1200)  # 父块大小
        self.CHILD_CHUNK_SIZE = self.config.getint("retrieval", "child_chunk_size", fallback=300)     # 子块大小
        self.CHUNK_OVERLAP = self.config.getint("retrieval", "chunk_overlap", fallback=50)            # 相邻块重叠字数
        self.RETRIEVAL_K = self.config.getint("retrieval", "retrieval_k", fallback=3)                 # 每路召回数量
        self.CANDIDATE_M = self.config.getint("retrieval", "candidate_m", fallback=2)                 # 重排后最终保留数

        # ===================== 应用业务配置 =====================
        # eval：把字符串 '["ai","java",...]' 解析成真正的 Python 列表
        # （省事写法；生产环境建议用 json.loads，避免 eval 的安全风险）
        self.VALID_SOURCES = eval(
            self.config.get("app", "valid_sources", fallback='["ai","java","test","ops","bigdata"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get("app", "customer_service_phone", fallback="13800138001")


# ============================================================
# 单例模式：模块被 import 多次，Config() 也只执行一次。
# 其它文件写 `from base.config import config` 就能直接用这个对象。
# ============================================================
config = Config()

# 直接运行本文件可以自检配置读取是否正常：python -m base.config
if __name__ == '__main__':
    conf = Config()
    print("MySQL:", conf.MYSQL_HOST, conf.MYSQL_USER)
    print("Redis:", conf.REDIS_HOST, conf.REDIS_PORT)
    print("日志路径:", conf.LOG_FILE)
    print("Milvus:", conf.MILVUS_COLLECTION_NAME)
