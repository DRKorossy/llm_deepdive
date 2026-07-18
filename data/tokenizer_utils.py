from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        strnums = [str(num) for num in numbers]
        return [self.tokenise_left_to_right_greedy(num, vocab) for num in strnums]

    def tokenise_left_to_right_greedy(self, text: str, vocab: Dict[str, int]) -> List[str]:
        l = len(text)
        idx = 0
        tokens = []
        while idx < l:
            max_str_in_dict = None
            for strlen in range(l - idx, 0, -1):
                substring = text[idx : idx + strlen]
                if substring in vocab:
                    max_str_in_dict = substring
                    break
            if max_str_in_dict is None:
                tokens.append(text[idx])
                idx += 1
            else:
                tokens.append(max_str_in_dict)
                idx += len(max_str_in_dict)
        return tokens

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self.tokenise_left_to_right_greedy(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        return round(self.count_tokens(text, vocab) / len(text.split()), 4)
