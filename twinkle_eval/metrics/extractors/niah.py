"""NIAH (Needle in a Haystack) Extractor。

NIAH 評測不需要從 response 中提取特定格式，直接 pass-through 原始回答。
Scorer 會對完整 response 做 substring match。
"""

from typing import Any, Optional

from twinkle_eval.core.abc import Extractor


class NIAHExtractor(Extractor):
    """NIAH Extractor — pass-through 原始 response。"""

    def get_name(self) -> str:
        return "niah"

    def extract(self, raw: Optional[Any]) -> Optional[Any]:
        """直接回傳原始 response，不做任何提取。"""
        return raw
