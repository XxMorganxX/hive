import re
from collections import Counter


def top_k_words(text: str, k: int) -> list[tuple[str, int]]:
    counts = Counter(re.findall(r"[a-z]+", text.lower()))
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]
