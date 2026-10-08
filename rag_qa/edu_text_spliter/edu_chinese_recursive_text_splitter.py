# -*- coding: utf-8 -*-
"""
rag_qa/edu_text_spliter/edu_chinese_recursive_text_splitter.py —— 中文递归文本分割器
================================================================

【为什么 RAG 要把长文档"切块（chunk）"？】
   1. 大模型上下文有长度限制，整本书塞不进去；
   2. 检索要"精准"：用户问一个小知识点，召回 100 字的相关段落
      比召回 20 页整章更准、更省 token；
   3. 块太大 → 噪声多、检索不准；块太小 → 语义不完整。
   所以"怎么切"是 RAG 效果的关键步骤之一。

【LangChain 自带 RecursiveCharacterTextSplitter 的思想】
   按优先级尝试分隔符：先按段落分，段落还太大就按句子分，
   句子还大就按逗号分……直到块大小小于 chunk_size。
   这叫"递归分割"，能尽量保证"语义完整"。

【本文件在它基础上改了什么？】
   默认 RecursiveCharacterTextSplitter 的分隔符是为英文设计的
   （["\n\n", "\n", " ", ""]，中文标点一个没有），
   这里重写了分隔符列表，加入中文句号、感叹号、问号、分号、逗号，
   所以叫 ChineseRecursiveTextSplitter（中文递归分割器）。

【关键参数】
   chunk_size    ：每块最大长度（本项目父块 1200 / 子块 300，见 config.ini）
   chunk_overlap ：相邻块重叠长度。为什么要有重叠？
                   如果一个知识点正好被切在边界处，
                   重叠后前后两块都保留它，检索时不会丢信息。
   keep_separator：切完后是否保留分隔符（中文句号保留着更自然）。
"""
import re
from typing import List, Optional, Any
# 继承 LangChain 的递归分割器（父类实现主流程，子类只改分隔符策略）
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """
    按正则分隔符切分文本的工具函数。
    "from_end" 的由来：这样切出来的块，分隔符会跟在"前一块的结尾"，
    而不是新块的开头（对中文来说读起来更自然）。

    参数：
        text:          待切文本
        separator:     分隔符（正则表达式）
        keep_separator:是否保留分隔符
    返回：切好的字符串列表（已去掉空串）。
    """
    # 有分隔符的情况
    if separator:
        if keep_separator:
            # 正则里加括号 → re.split 会把"分隔符本身"也保留在结果里，
            # 例如 re.split("(。)", "你好。世界") → ['你好', '。', '世界']
            _splits = re.split(f"({separator})", text)
            # 把"正文片段"和"它后面的分隔符"两两拼接：
            # zip(_splits[0::2], _splits[1::2]) 取出 (片段, 分隔符) 对
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 如果切完是奇数个，说明最后还剩一个"没有分隔符跟着的尾巴"，补上
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            # 不保留分隔符：直接切，分隔符丢弃
            splits = re.split(separator, text)
    else:
        # 没有可用分隔符了（最后兜底）：逐字符切开，保证一定能小于 chunk_size
        splits = list(text)
    # 过滤掉空串，返回
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """中文递归分割器：继承父类，只定制"中文分隔符列表"。"""

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """创建一个中文分割器。"""
        # keep_separator 交给父类保存（父类合并、切分时要用）
        super().__init__(keep_separator=keep_separator, **kwargs)
        # 分隔符优先级列表（从"最语义完整"到"最细碎"逐级降级）：
        #   1. 两个换行 = 段落边界
        #   2. 一个换行 = 行边界
        #   3. 中文句号/感叹号/问号 = 句子边界
        #   4. 英文句号/感叹号/问号 + 空格
        #   5. 中文/英文分号
        #   6. 中文/英文逗号
        # 传参了就用手动传的，否则用这套默认值
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s"
        ]
        # 标记分隔符是正则（True），父类/本类用 re.search 处理
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        核心切分逻辑（对父类同名方法的覆盖/重写）。

        流程：
          1. 从高优先级分隔符里挑"出现在文本中的第一个"来切；
          2. 切出的片段：
             - 小于 chunk_size → 先攒着（好片段）
             - 大于 chunk_size → 递归用更低级的分隔符继续切
          3. 攒的片段够多时，调父类 _merge_splits 合并成接近 chunk_size 的块
             （合并时会按 chunk_overlap 做重叠）。
        """
        final_chunks = []  # 最终输出的块
        # 默认拿最低优先级的分隔符兜底
        separator = separators[-1]
        new_separators = []  # 当前分隔符后面还有哪些更低级的分隔符
        # 从最高优先级开始，找"第一个真的出现在文本里的分隔符"
        for i, _s in enumerate(separators):
            # 如果分隔符不是正则，就转义它（防止里面的特殊字符被当正则）
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s  # 空串 = 逐字符切，直接选中
                break
            if re.search(_separator, text):  # 文本里能找到这个分隔符
                separator = _s                # 就选它
                new_separators = separators[i + 1:]  # 记录更细的分隔符备用
                break

        # 用选中的分隔符切分
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # 开始合并/递归切分更长的片段
        _good_splits = []  # 攒"已经足够小"的片段
        # 合并时用什么连接符：保留分隔符的话，片段自己已经带分隔符了，用 ""；
        # 不保留的话，用原分隔符拼回去
        _separator = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size:
                # 片段小于 chunk_size → 攒着，等会儿和邻居合并
                _good_splits.append(s)
            else:
                # 片段还太大 → 先把攒着的合并输出
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    # 没有更细的分隔符了，这块只能这么大（兜底）
                    final_chunks.append(s)
                else:
                    # 递归：用更细的分隔符继续切这个片段
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        # 收尾：把最后攒着的好片段也合并输出
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        # 清理：把连续 2 个以上换行压成 1 个，去掉每块首尾空白，丢弃空块
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


# ============================================================
# 【单独调试】python -m rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter
# 用一段中文贸易报告验证切块效果：
# chunk_size=150 表示块最大 150 字，chunk_overlap=10 表示相邻块重叠 10 字。
# ============================================================
if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,       # 保留中文标点
        is_separator_regex=True,   # 分隔符按正则解释
        chunk_size=150,            # 每块最大 150 字
        chunk_overlap=10           # 相邻块重叠 10 字
    )
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
    ]
    # 逐段测试切块并打印
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
