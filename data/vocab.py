from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        uniques = sorted(set(text))
        stoi = {c:idx for idx, c in enumerate(uniques)}
        itos = {idx:c for c, idx in stoi.items()}
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        return [stoi[c] for c in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        return "".join([itos[idx] for idx in ids])
