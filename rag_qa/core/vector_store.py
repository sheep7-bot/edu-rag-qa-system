# -*- coding: utf-8 -*-
"""
rag_qa/core/vector_store.py —— 向量存储引擎（Milvus + BGE-M3 + 重排序）
================================================================

【这个文件在 RAG 架构里的位置】
   离线建库：add_documents()（文档 → 向量 → 存 Milvus）
   在线检索：hybrid_search_with_reranker()（问题 → 向量 → 混合检索 → 重排 → 父块）

【技术 1：向量数据库 Milvus】
   普通数据库按"等于/大于"查数据；向量库按"语义相似度"查数据。
   比如搜"学费多少"，能召回"课程价格是多少"——因为两个句子的向量很接近。
   Milvus 是最流行的开源向量数据库之一。
   怎么打开/启动（本地单机版）：
     1. Docker 方式（推荐）：
          下载 docker-compose.yml（Milvus 官网 standalone 版），执行：
              docker compose up -d
          服务端口：19530（SDK 连接用）、9091（管理页面）
     2. 轻量替代：pip install milvus-lite，URI 直接写本地 .db 文件路径，无需 Docker。
     3. 验证：pip install pymilvus 后运行本文件的 __main__ 能连通即成功。
   ⚠️ 本项目 config.ini 里 uri=http://localhost:19530，用的是 Docker 独立版。

【技术 2：BGE-M3 嵌入模型（把文字变成向量）】
   BGE-M3（BAAI 出品）一次能输出两种向量（多向量混合检索的核心）：
     - dense（稠密向量）：1024 维浮点数，表达"整体语义"，擅长同义改写匹配；
     - sparse（稀疏向量）：词项权重（类似 TF-IDF），表达"关键词命中"，
       擅长专有名词、编号、缩写等精确匹配。
   本项目两种都用 → 叫"混合检索（Hybrid Search）"。
   模型文件需提前下载到本地：E:\\Develop\\embedding\\bge-m3
   （下载方式：modelscope 或 huggingface 搜索 BAAI/bge-m3）

【技术 3：BGE-reranker-large 重排序模型（精排）】
   检索分两阶段（工业界标准做法）：
     粗排：向量检索从全库快速捞出候选（快，但精度一般）；
     精排：用 CrossEncoder 把"问题-文档"成对过一遍模型打分（慢，但准）。
   重排序只对少量候选做，所以整体速度可接受、精度明显提升。
   模型路径：E:\\Develop\\embedding\\bge-reranker-large

【技术 4：索引类型（建库时的加速结构）】
   - dense_vector → IVF_FLAT：倒排文件索引。先把向量聚成 nlist=128 个簇，
     检索时只扫最近的几个簇，避免全表扫描。
   - sparse_vector → SPARSE_INVERTED_INDEX：稀疏倒排索引，按非零维度组织。
   - metric_type="IP"：内积（Inner Product），BGE 系列向量归一化后
     内积=余弦相似度，越大越相似。
"""
import hashlib  # 生成 MD5 哈希当主键
from milvus_model.hybrid import BGEM3EmbeddingFunction  # BGE-M3 官方封装（稠密+稀疏）
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker  # Milvus SDK
from sentence_transformers import CrossEncoder  # 重排序模型（CrossEncoder 架构）
from langchain_core.documents import Document   # LangChain 文档对象

from base.config import Config
from base.logger import logger

# 加载项目全局配置
conf = Config()


class VectorStore:
    """
    向量存储管理类。
    封装：Milvus 连接、BGE-M3 模型初始化、集合自动创建、索引构建、混合检索。
    """

    def __init__(self):
        """
        构造方法：执行完整初始化流程。
        1. 加载本地 BGE-M3 混合嵌入模型
        2. 建立 Milvus 连接，自动创建目标数据库并切换
        3. 检测集合，不存在则自动创建 Schema + 稠密/稀疏索引
        4. 加载重排序模型
        """
        # ---------------------- 1. 初始化 BGE-M3 混合嵌入模型 ----------------------
        # 本地 BGE-M3 模型存放路径（按你机器实际位置修改）
        bge_m3_model_path = r"E:\Develop\embedding\bge-m3"

        # 实例化混合向量模型：调用一次，同时得到稠密向量和稀疏向量
        self.embedding = BGEM3EmbeddingFunction(model_name_or_path=bge_m3_model_path)
        logger.info("BGE-M3 混合嵌入模型初始化完成")

        # ---------------------- 2. Milvus 连接 + 数据库管理 ----------------------
        # 创建 Milvus 客户端连接（uri 如 http://localhost:19530）
        self.client = MilvusClient(
            uri=conf.MILVUS_URI
        )
        logger.info(f"Milvus 基础连接成功")

        target_db = conf.MILVUS_DATABASE_NAME
        # 列出已有数据库
        db_list = self.client.list_databases()
        # 判断目标数据库是否存在，不存在自动新建（若权限不足会抛异常）
        if target_db not in db_list:
            self.client.create_database(target_db)
            logger.info(f"数据库 {target_db} 不存在，自动创建完成")

        # 切换当前客户端到目标数据库（后续操作都在这个库里）
        self.client.using_database(target_db)
        logger.info(f"已切换至数据库: {target_db}")

        # ---------------------- 3. 自动创建向量集合与索引 ----------------------
        self.create_collection()

        # ---------------------- 4. 初始化重排序模型 ----------------------
        bge_reranker_path = r"E:\Develop\embedding\bge-reranker-large"
        # CrossEncoder：把 (问题, 文档) 拼成一对输入打分，精度高于普通向量相似度
        self.reranker = CrossEncoder(bge_reranker_path)
        logger.info("BGE-Reranker 重排序模型初始化完成")


    def create_collection(self):
        """
        创建 Milvus 向量集合（相当于关系数据库里的"建表"）。
        逻辑：集合不存在才创建，避免重复初始化。
        内容：定义数据表 Schema、字段含义、稠密向量索引、稀疏向量索引。
        """
        coll_name = conf.MILVUS_COLLECTION_NAME
        if not self.client.has_collection(coll_name):
            # 创建表结构：auto_id=False（主键自己给），enable_dynamic_field=True（允许额外字段）
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)

            # ---- 字段 1：主键 ----
            # 文本切块唯一标识（MD5 生成，见 add_documents）
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
                description="文本切块唯一ID"
            )
            # ---- 字段 2：子块文本 ----
            # 用于检索匹配的最小文本单元（对应"子块"）
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                description="文本子块内容"
            )
            # ---- 字段 3：稠密向量 ----
            # BGE-M3 输出，捕捉整体语义；维度由模型决定（bge-m3 是 1024）
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embedding.dim['dense'],  # 自动取模型维度，不写死
                description="BGE-M3稠密语义向量"
            )
            # ---- 字段 4：稀疏向量 ----
            # BGE-M3 输出，捕捉关键词/词项特征
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="BGE-M3稀疏词向量"
            )
            # ---- 字段 5：父块 ID ----
            # 子块归属的父文档 ID（父子分块关联键）
            schema.add_field(
                field_name="parent_id",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="父文本块ID，实现父子块关联"
            )
            # ---- 字段 6：父块完整原文 ----
            # 检索命中后，实际喂给大模型的是这段完整长文本
            schema.add_field(
                field_name="parent_content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                description="父文本完整原文"
            )
            # ---- 字段 7：学科标签 ----
            # 用于检索时条件过滤（expr="source == 'ai'"）
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=50,
                description="学科/数据源名称，用于检索过滤"
            )
            # ---- 字段 8：入库时间戳 ----
            # 用于数据生命周期管理（比如清理旧数据）
            schema.add_field(
                field_name="timestamp",
                datatype=DataType.VARCHAR,
                max_length=50,
                description="文档入库时间戳"
            )

            # 索引配置容器：可以一次性批量定义多个向量索引
            index_params = self.client.prepare_index_params()

            # ========== 稠密向量索引：IVF_FLAT 倒排索引 ==========
            # nlist：聚类分组数量（把向量聚成 128 个簇），用于加速检索。
            # 检索时只扫最接近的 nprobe 个簇（见 hybrid_search 的 nprobe=10）。
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )

            # ========== 稀疏向量索引：稀疏倒排索引 ==========
            # drop_ratio_build：建索引时裁剪每条向量权重最低 20% 的弱特征，减少内存占用
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 执行创建集合 + 构建索引
            self.client.create_collection(
                collection_name=coll_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"向量集合创建成功: {coll_name}")
        else:
            logger.info(f"向量集合已存在，无需重复创建: {coll_name}")


    def add_documents(self, documents):
        """
        将文档（子块）向量化后存入 Milvus（离线建库的核心方法）。

        参数 documents：document_processor.process_documents() 产出的子块列表，
        每个 doc 的 metadata 里必须有 parent_id / parent_content / source / timestamp。
        """
        # 获取文档的文本内容（子块文本列表）
        texts = [doc.page_content for doc in documents]

        # 用 BGE-M3 批量生成嵌入：
        #   embeddings["dense"]  → 稠密向量矩阵（每行一个 1024 维向量）
        #   embeddings["sparse"] → 稀疏向量对象列表（内部是 列索引/值 两个数组）
        embeddings = self.embedding(texts)

        data = []  # 待插入的数据行
        # 遍历每个文档（带上索引 i，用来取对应位置的向量）
        for i, doc in enumerate(documents):
            # 生成文档内容的 MD5 哈希值，作为唯一 ID：
            #   同一段文本重复入库会覆盖而不是产生重复记录（幂等）。
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

            # 把 BGE-M3 的稀疏向量转成 Milvus 需要的"字典"格式 {维度下标: 权重}
            sparce_vector = {}
            sparse_emb = embeddings["sparse"][i]  # 第 i 条稀疏向量（csr 行）
            # sparse_emb.col 是非零维度的下标数组，.data 是对应权重值
            for index, val in zip(sparse_emb.col, sparse_emb.data):
                sparce_vector[int(index)] = float(val)

            # 组装一行数据（字段名必须和建表时的 schema 一致）
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings['dense'][i],   # 稠密向量
                "sparse_vector": sparce_vector,           # 稀疏向量（字典形式）
                "parent_id": doc.metadata["parent_id"],   # 父块 ID
                "parent_content": doc.metadata["parent_content"],  # 父块全文
                "source": doc.metadata.get("source", "unknown"),   # 学科
                "timestamp": doc.metadata.get("timestamp", "unknown"),  # 入库时间
            })

        if data:
            # upsert = update + insert：主键已存在则覆盖，不存在则插入
            self.client.upsert(collection_name=conf.MILVUS_COLLECTION_NAME, data=data)
            logger.info(f"已插入或更新 {len(data)} 个文档")

    def hybrid_search_with_reranker(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        混合检索（稠密+稀疏）+ 加权融合 + BGE 重排序 + 父块去重。

        :param query:         用户问题
        :param k:             每路初步召回数量（默认 3，config.ini 可配）
        :param source_filter: 学科过滤（如 "ai"，None 表示全库搜）
        :return: 重排后父块 Document 列表（最终给大模型的上下文）
        """
        # 1. 生成查询向量（稀疏+稠密都生成，和入库时用同一模型保证同空间）
        embedding_result = self.embedding([query])
        dense_q = embedding_result['dense'][0]   # 查询的稠密向量

        # 稀疏向量格式化：Milvus 的检索请求要求字典形式 {下标: 权重}
        sparse_q = {}
        row = embedding_result['sparse'][[0]]    # 取第 0 条（外面套列表保持批量维度）
        for idx, val in zip(row.indices, row.data):
            sparse_q[idx] = val

        # 2. 过滤条件（Milvus 表达式语法；不传 source_filter 则空串=不过滤）
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 3. 构造两路独立检索请求
        #    稠密路：整体语义匹配
        dense_req = AnnSearchRequest(
            data=[dense_q],                 # 查询向量
            anns_field="dense_vector",      # 在哪个字段上检索
            param={"metric_type": "IP", "params": {"nprobe": 10}},  # 扫 10 个簇
            limit=k,                        # 召回数量
            expr=filter_expr                # 过滤条件
        )

        #    稀疏路：关键词级匹配
        sparse_req = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=k,
            expr=filter_expr
        )

        # WeightedRanker(稠密权重, 稀疏权重)：
        #   权重比 0.7 : 1.0 → 本项目更看重关键词精确匹配（教育场景专有名词多）。
        #   两路分数先各自归一化，再加权求和，得到融合排序。
        ranker = WeightedRanker(0.7, 1.0)

        # 4. 执行混合检索
        #    返回值是"按查询分组的二维列表"，[0] 取第一个查询的结果
        results = self.client.hybrid_search(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            reqs=[dense_req, sparse_req],   # 两路请求
            ranker=ranker,                  # 融合策略
            limit=k,                        # 最终返回条数
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        # 5. 把 Milvus 命中结果转成 LangChain Document
        chunks = [self.__doc_from_hit(hit["entity"]) for hit in results]

        # 6. 父块去重：多个子块可能属于同一个父块，只保留一份（避免上下文重复）
        parent_docs = self.__get_unique_parent_docs(chunks)

        # 7. 重排序：候选 ≥2 条时才值得精排
        if len(parent_docs) >= 2:
            # 把 (query, 父块全文) 组成对，交给 CrossEncoder 打分
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs).flatten()
            # 分数和文档绑定成 (0.9, 文档A), (0.5, 文档B) 再按分数降序排
            sorted_docs = sorted(zip(scores, parent_docs), reverse=True)
            # 拆出排好序的文档列表
            parent_docs = [doc for score, doc in sorted_docs]

        # 返回最终候选集（默认取前 CANDIDATE_M=2 条）
        return parent_docs[:conf.CANDIDATE_M]

    def __doc_from_hit(self, entity):
        """Milvus 命中结果（entity 字典）转为 LangChain Document 对象。"""
        return Document(
            page_content=entity.get("text"),  # 先用子块文本（重排后会被父块替换）
            metadata={
                "parent_id": entity.get("parent_id"),
                "parent_content": entity.get("parent_content"),
                "source": entity.get("source"),
                "timestamp": entity.get("timestamp")
            }
        )

    def __get_unique_parent_docs(self, chunks):
        """
        从命中的子块中提取"去重后的父块"。
        原理：子块 metadata 里带着 parent_content，
        用 set 记录已见过的父块全文，重复的直接跳过。
        """
        seen = set()
        unique_docs = []
        for chunk in chunks:
            content = chunk.metadata.get("parent_content")
            if content and content not in seen:
                seen.add(content)
                unique_docs.append(
                    Document(page_content=content, metadata=chunk.metadata)  # 用父块全文建新 Document
                )
        return unique_docs


# ============================================================
# 【单独调试：离线建库 + 检索测试】
# 运行：python -m rag_qa.core.vector_store
# 流程：初始化(自动建库建集合) → 处理 data 目录文档 → 向量化入库 → 试检索
# ⚠️ 第一次运行前确认：
#    1. Milvus 已启动（docker compose up -d 或 milvus-lite）；
#    2. BGE-M3、BGE-reranker-large 模型已下载到对应路径。
# ============================================================
if __name__ == '__main__':
    # 测试：初始化向量存储（自动创建集合与索引）
    vs = VectorStore()
    # 加载拆分文档
    from document_processor import process_documents
    # 直接传入目录路径
    dir_path = r"E:\py_demo\code06\integrated_qa_system\rag_qa\data"
    chunks = process_documents(dir_path)
    # 向量化存储
    vs.add_documents(chunks)

    # 查询测试
    docs = vs.hybrid_search_with_reranker(query="⼤语⾔模型 (LLM) 背景")
    for doc in docs:
        print(doc)
        print("--------------")
