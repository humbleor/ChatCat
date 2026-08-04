"""企业级文本净化器。

分层、确定性、幂等的文本清洗流水线：
- L1 字符层：NFC 规范化、C0/C1 控制符、非字符、孤立代理项、U+FFFD、不可见字符。
- L2 规范化层：全角 ASCII 折叠为半角、各类"假空格"归一半角空格。
- L3 空白层：行尾统一 LF、行内连续空白折叠、连续空行压缩（保住 \n\n 段落边界）。
- L4 私有区：PUA 默认清除，可用 strip_pua 关闭。

设计约束：所有步骤均为确定性查表，不含任何启发式"还原/猜测"，保证幂等。
"""

from __future__ import annotations

import re
import unicodedata

# L1 —— C0（保留 \t \n \r）+ C1（0x80-0x9F）+ DEL
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")
# L1 —— 不可见字符：软连字符、零宽(含 ZWNJ/ZWJ)、LRM/RLM、bidi 全集合(含 isolates)、word joiner、BOM
_INVISIBLE_CHAR_RE = re.compile(
    r"[\u00ad\u061c\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]"
)
# L1 —— 非字符
_NONCHARACTER_RE = re.compile(r"[\ufdd0-\ufdef\ufffe\uffff]")
# L1 —— 解码丢失标记（U+FFFD）
_REPLACEMENT_CHAR_RE = re.compile(r"\ufffd")
# L4 —— PUA 主平面
_PUA_RE = re.compile(r"[\ue000-\uf8ff]")
# L4 —— 补充 PUA 平面（15/16 平面）
_SUPPLEMENTARY_PUA_RE = re.compile(r"[\U000f0000-\U0010fffd]")

# L2 —— 全角 ASCII 块 U+FF01..FF5E → 半角 U+0021..007E（确定性查表）
_FULLWIDTH_TRANS = str.maketrans(
    {0xFF01 + i: 0x0021 + i for i in range(0x5E)}
)
# L2 —— 各类"假空格" → 半角空格
_SPACES_TRANS = str.maketrans(
    {0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x3000: " "}
)

# L3 —— 行尾统一
_LINE_BREAK_RE = re.compile(r"\r\n?|[\n\u2028\u2029]")
# L3 —— 行内连续空白折叠
_SPACE_RUN_RE = re.compile(r"[ \t]+")
# L3 —— 连续 3+ 换行压缩为 2（保住 \n\n 段落边界）
_BLANK_LINE_RE = re.compile(r"\n{3,}")


def _drop_surrogates(text: str) -> str:
    if not any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return text
    return "".join(c for c in text if not 0xD800 <= ord(c) <= 0xDFFF)


def sanitize_text(text: str, *, strip_pua: bool = True) -> str:
    """全流水线。文档内容与 query 共用，保证索引/查询特征空间对称。"""
    if not text:
        return ""
    # L1 字符层
    t = unicodedata.normalize("NFC", text)
    t = _CONTROL_CHAR_RE.sub("", t)
    t = _NONCHARACTER_RE.sub("", t)
    t = _drop_surrogates(t)
    t = _REPLACEMENT_CHAR_RE.sub("", t)
    t = _INVISIBLE_CHAR_RE.sub("", t)
    # L2 规范化层
    t = t.translate(_FULLWIDTH_TRANS)
    t = t.translate(_SPACES_TRANS)
    # L3 空白层
    t = _LINE_BREAK_RE.sub("\n", t)
    t = _SPACE_RUN_RE.sub(" ", t)
    t = _BLANK_LINE_RE.sub("\n\n", t)
    # L4 私有区
    if strip_pua:
        t = _PUA_RE.sub("", t)
        t = _SUPPLEMENTARY_PUA_RE.sub("", t)
    return t


def sanitize_metadata(value: str) -> str:
    """元数据轻量版：字符级 scrub + NFC；绝不折叠空白、不折叠全角，保护文件名/路径。"""
    if not value:
        return ""
    t = unicodedata.normalize("NFC", value)
    t = _CONTROL_CHAR_RE.sub("", t)
    t = _NONCHARACTER_RE.sub("", t)
    t = _drop_surrogates(t)
    t = _REPLACEMENT_CHAR_RE.sub("", t)
    t = _INVISIBLE_CHAR_RE.sub("", t)
    t = _PUA_RE.sub("", t)
    return t
